from transformers import Trainer, TrainingArguments, AutoTokenizer, EsmForMaskedLM, TrainerCallback
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pandas as pd
import torch
from torch.optim import AdamW
# import wandb
import numpy as np
import argparse
from torch.distributions.categorical import Categorical
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ================== 1. 定义命令行参数 ==================
parser = argparse.ArgumentParser(description="PepMLM Training Script")
parser.add_argument("--train_file", type=str, required=True,
                    help="Path to the training CSV file")
parser.add_argument("--val_file", type=str, required=True,
                    help="Path to the validation CSV file")
parser.add_argument("--test_file", type=str, required=True,
                    help="Path to the training CSV file")
parser.add_argument("--output_dir", type=str, default="./output_final/",
                    help="Directory to save model checkpoints")
parser.add_argument("--logging_dir", type=str, default="./logs",
                    help="Directory to save training logs")
parser.add_argument("--dataset", type=str, default="pepbench",
                    help="Directory to save model checkpoints")
args = parser.parse_args()

class ProteinDataset(Dataset):
    def __init__(self, file, tokenizer):
        data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.proteins = data["Receptor Sequence"].tolist()
        self.peptides = data["Binder"].tolist()

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        peptide_seq = self.peptides[idx]

        masked_peptide = '<mask>' * len(peptide_seq)
        complex_seq = protein_seq + masked_peptide

        # Tokenize and pad the complex sequence
        complex_input = self.tokenizer(complex_seq, return_tensors="pt", padding="max_length", max_length = 552, truncation=True)

        input_ids = complex_input["input_ids"].squeeze()
        attention_mask = complex_input["attention_mask"].squeeze()

        # Create labels
        label_seq = protein_seq + peptide_seq
        labels = self.tokenizer(label_seq, return_tensors="pt", padding="max_length", max_length = 552, truncation=True)["input_ids"].squeeze()

        # Set non-masked positions in the labels tensor to -100
        labels = torch.where(input_ids == self.tokenizer.mask_token_id, labels, -100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
model_name = "esm2_t33_650M_UR50D"
model = EsmForMaskedLM.from_pretrained("facebook/" + model_name)
tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)
print(f"load {model_name}")
lr = 0.0007984276816171436

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs = 10,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 16,
    warmup_steps = 501,
    logging_dir=args.logging_dir,
    logging_steps=10,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_strategy='epoch',
    metric_for_best_model='eval_loss',
    save_total_limit = 1,
    gradient_accumulation_steps=2,
    report_to="none"
)

train_dataset = ProteinDataset(args.train_file, tokenizer)
val_dataset = ProteinDataset(args.val_file, tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(AdamW(model.parameters(), lr=lr), None),
)

trainer.train()





def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    sequence = protein_seq + binder_seq
    original_input = tokenizer.encode(sequence, return_tensors='pt').to(model.device)
    length_of_binder = len(binder_seq)

    masked_inputs = original_input.repeat(length_of_binder, 1)
    positions_to_mask = torch.arange(-length_of_binder - 1, -1, device=model.device)
    masked_inputs[torch.arange(length_of_binder), positions_to_mask] = tokenizer.mask_token_id

    labels = torch.full_like(masked_inputs, -100)
    labels[torch.arange(length_of_binder), positions_to_mask] = original_input[0, positions_to_mask]

    with torch.no_grad():
        outputs = model(masked_inputs, labels=labels)
        loss = outputs.loss

    avg_loss = loss.item()
    pseudo_perplexity = np.exp(avg_loss)
    return pseudo_perplexity

def compute_pseudo_perplexity2(model, tokenizer, protein_seq, binder_seq):
    sequence = protein_seq + binder_seq
    tensor_input = tokenizer.encode(sequence, return_tensors='pt').to(model.device)
    total_loss = 0

    for i in range(-len(binder_seq)-1, -1):
        masked_input = tensor_input.clone()
        masked_input[0, i] = tokenizer.mask_token_id
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        with torch.no_grad():
            outputs = model(masked_input, labels=labels)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(binder_seq)
    pseudo_perplexity = np.exp(avg_loss)
    return pseudo_perplexity

def generate_peptide_for_single_sequence(protein_seq, peptide_length=15, top_k=3, num_binders=4):
    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    binders_with_ppl = []

    for _ in range(num_binders):
        masked_peptide = '<mask>' * peptide_length
        input_sequence = protein_seq + masked_peptide
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(** inputs).logits
        mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]

        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_indices = Categorical(probabilities).sample()
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl

def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4):
    if isinstance(input_seqs, str):
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
        return pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])
    elif isinstance(input_seqs, list):
        results = []
        for seq in input_seqs:
            binders = generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
            for binder, ppl in binders:
                results.append([seq, binder, ppl])
        return pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])

# ----------------------
# 2. 加载模型和tokenizer
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")

subdirs = [d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))]
if len(subdirs) != 1:
    raise ValueError(f"Expected exactly one checkpoint folder in {args.output_dir}, found {len(subdirs)}")
checkpoint_path = os.path.join(args.output_dir, subdirs[0])
model = EsmForMaskedLM.from_pretrained(
    checkpoint_path
).to(device)

model.eval()  # 切换到评估模式

# ----------------------
# 3. 处理test.csv
# ----------------------
#读取输入文件 生成肽
input_df = pd.read_csv(args.test_file)  # 替换为你的test.csv路径

# 存储所有结果
all_results = []
from tqdm import tqdm
topk = 3
print(f"\n正在处理 topk = {topk}")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats() 
model_loading_memory = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"模型权重占用显存: {model_loading_memory:.2f} MB")
for idx, row in tqdm(input_df.iterrows()):
    torch.cuda.reset_peak_memory_stats()
    print(idx)
    receptor_seq = row["Receptor Sequence"]  # 受体序列
    peptide_length = row["Sequence Length"]   # 肽段长度（从原数据读取）
    num_binders = 20                         # 每个受体生成20个肽段
    top_k = topk                                 # 可根据需要调整

    # 生成肽段
    try:
        # 调用生成函数，指定当前行的长度和生成数量
        generated_df = generate_peptide(
            input_seqs=receptor_seq,
            peptide_length=peptide_length,
            top_k=top_k,
            num_binders=num_binders
        )
        
        # 3. 统计该次生成的全量峰值
        total_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # 4. 计算纯推理开销 (峰值 - 模型基准)
        inference_net_peak = total_peak - model_loading_memory
        
        print(f"Index {idx} | 长度 {peptide_length}:")
        print(f"  - 总显存峰值: {total_peak:.2f} MB")
        print(f"  - 额外生成开销: {inference_net_peak:.2f} MB")
        
        
    except Exception as e:
        print(f"处理第{idx}行时出错: {e}")
        continue

    # 将生成的肽段与原数据的其他列关联
    # 复制原行的信息（如Receptor Sequence Length等）
    row_info = row.drop(["Binder", "Sequence Length"]).to_dict()  # 排除不需要的列
    # 为每个生成的肽段添加原行信息
    for _, gen_row in generated_df.iterrows():
        result = {
            **row_info,  # 原行的其他信息（如Receptor Sequence、Receptor Sequence Length等）
            "Generated Binder": gen_row["Binder"],  # 生成的肽段
            "Sequence Length": peptide_length,      # 肽段长度（原数据指定）
            "Pseudo Perplexity": gen_row["Pseudo Perplexity"]  # 伪困惑度
        }
        all_results.append(result)

# 转换为DataFrame并保存
output_df = pd.DataFrame(all_results)
output_df.to_csv(f"{args.output_dir}/{args.dataset}_test.csv", index=False)  # 输出文件
print(f"生成完成，结果保存至 {args.output_dir}/{args.dataset}_test.csv")
