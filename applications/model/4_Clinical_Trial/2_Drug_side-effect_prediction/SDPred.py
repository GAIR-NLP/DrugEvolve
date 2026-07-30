import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvNCF(nn.Module):
    def __init__(self, drugs_dim, sides_dim, embed_dim, bathsize, dropout1=0.8, dropout2=0.8):
        super(ConvNCF, self).__init__()

        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.batchsize = bathsize
        self.drug_dim = self.drugs_dim//10
        self.side_dim = self.sides_dim//4
        # self.side_dim = side_dim
        # self.drug_dim = drug_dim
        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.drugs_layer = nn.Linear(self.drugs_dim, self.embed_dim)
        self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.sides_layer = nn.Linear(self.sides_dim, self.embed_dim)
        self.sides_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.sides_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        self.drug_layer1 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer2 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer3 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer3_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer4 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer4_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer5 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer5_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer6 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer6_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer7 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer7_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer8 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer8_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer9 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer9_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer10 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer10_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.side_layer1 = nn.Linear(self.side_dim, self.embed_dim)
        self.side_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.side_layer2 = nn.Linear(self.side_dim, self.embed_dim)
        self.side_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.side_layer3 = nn.Linear(self.side_dim, self.embed_dim)
        self.side_layer3_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.side_layer4 = nn.Linear(self.side_dim, self.embed_dim)
        self.side_layer4_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.drug1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug3_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug4_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug5_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug6_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug7_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug8_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug9_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug10_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.side1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.side2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.side3_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.side4_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.num_class = 2
        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.number_map = 10 * 4
        self.cnn_interaction = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(self.number_map, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )
        self.total_layer = nn.Linear((self.channel_size * 4 + 2 * self.embed_dim), self.channel_size * 4)
        self.total_bn = nn.BatchNorm1d((self.channel_size * 4 + 2 * self.embed_dim), momentum=0.5)
        self.classifier = nn.Linear(self.channel_size * 4, 1)
        self.con_layer = nn.Linear(self.channel_size * 4, 1)

    def forward(self, drug_features, side_features, device):

        x_drugs = F.relu(self.drugs_bn(self.drugs_layer(drug_features.to(device))), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
        x_drugs = self.drugs_layer_1(x_drugs)

        x_sides = F.relu(self.sides_bn(self.sides_layer(side_features.to(device))), inplace=True)
        x_sides = F.dropout(x_sides, training=self.training, p=self.dropout1)
        x_sides = self.sides_layer_1(x_sides)


        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10 = drug_features.chunk(10, 1)
        side1, side2, side3, side4 = side_features.chunk(4, 1)


        x_drug1 = F.relu(self.drug1_bn(self.drug_layer1(drug1.to(device))), inplace=True)
        x_drug1 = F.dropout(x_drug1, training=self.training, p=self.dropout1)
        x_drug1 = self.drug_layer1_1(x_drug1)

        x_drug2 = F.relu(self.drug2_bn(self.drug_layer2(drug2.to(device))), inplace=True)
        x_drug2 = F.dropout(x_drug2, training=self.training, p=self.dropout1)
        x_drug2 = self.drug_layer2_1(x_drug2)

        x_drug3 = F.relu(self.drug3_bn(self.drug_layer3(drug3.to(device))), inplace=True)
        x_drug3 = F.dropout(x_drug3, training=self.training, p=self.dropout1)
        x_drug3 = self.drug_layer3_1(x_drug3)

        x_drug4 = F.relu(self.drug4_bn(self.drug_layer4(drug4.to(device))), inplace=True)
        x_drug4 = F.dropout(x_drug4, training=self.training, p=self.dropout1)
        x_drug4 = self.drug_layer4_1(x_drug4)

        x_drug5 = F.relu(self.drug5_bn(self.drug_layer5(drug5.to(device))), inplace=True)
        x_drug5 = F.dropout(x_drug5, training=self.training, p=self.dropout1)
        x_drug5 = self.drug_layer5_1(x_drug5)

        x_drug6 = F.relu(self.drug6_bn(self.drug_layer6(drug6.to(device))), inplace=True)
        x_drug6 = F.dropout(x_drug6, training=self.training, p=self.dropout1)
        x_drug6 = self.drug_layer6_1(x_drug6)

        x_drug7 = F.relu(self.drug7_bn(self.drug_layer7(drug7.to(device))), inplace=True)
        x_drug7 = F.dropout(x_drug7, training=self.training, p=self.dropout1)
        x_drug7 = self.drug_layer7_1(x_drug7)

        x_drug8 = F.relu(self.drug8_bn(self.drug_layer8(drug8.to(device))), inplace=True)
        x_drug8 = F.dropout(x_drug8, training=self.training, p=self.dropout1)
        x_drug8 = self.drug_layer8_1(x_drug8)

        x_drug9 = F.relu(self.drug9_bn(self.drug_layer9(drug9.to(device))), inplace=True)
        x_drug9 = F.dropout(x_drug9, training=self.training, p=self.dropout1)
        x_drug9 = self.drug_layer9_1(x_drug9)

        x_drug10 = F.relu(self.drug10_bn(self.drug_layer10(drug10.to(device))), inplace=True)
        x_drug10 = F.dropout(x_drug10, training=self.training, p=self.dropout1)
        x_drug10 = self.drug_layer10_1(x_drug10)

        drugs = [x_drug1, x_drug2, x_drug3, x_drug4, x_drug5, x_drug6, x_drug7, x_drug8, x_drug9, x_drug10]

        x_side1 = F.relu(self.side1_bn(self.side_layer1(side1.to(device))), inplace=True)
        x_side1 = F.dropout(x_side1, training=self.training, p=self.dropout1)
        x_side1 = self.side_layer1_1(x_side1)

        x_side2 = F.relu(self.side2_bn(self.side_layer2(side2.to(device))), inplace=True)
        x_side2 = F.dropout(x_side2, training=self.training, p=self.dropout1)
        x_side2 = self.side_layer2_1(x_side2)

        x_side3 = F.relu(self.side3_bn(self.side_layer3(side3.to(device))), inplace=True)
        x_side3 = F.dropout(x_side3, training=self.training, p=self.dropout1)
        x_side3 = self.side_layer3_1(x_side3)

        x_side4 = F.relu(self.side4_bn(self.side_layer4(side4.to(device))), inplace=True)
        x_side4 = F.dropout(x_side4, training=self.training, p=self.dropout1)
        x_side4 = self.side_layer4_1(x_side4)

        sides = [x_side1, x_side2, x_side3, x_side4]

        maps = []
        for i in range(len(drugs)):
            for j in range(len(sides)):
                maps.append(torch.bmm(drugs[i].unsqueeze(2), sides[j].unsqueeze(1)))

        interaction_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))

        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            interaction_map = torch.cat([interaction_map, interaction], dim=1)
        # interaction_map = interaction_map.view((-1, 1, self.embed_dim, self.embed_dim))
        feature_map = self.cnn_interaction(interaction_map)  # output: batch_size * 32 * 1 * 1
        # print(feature_map.size())
        h = feature_map.view((-1, 32*4))
        # print(h.size())
        # print(x_drugs.size())
        # print(x_sides.size())
        total = torch.cat((x_drugs, h, x_sides), dim=1)
        # print(total.size())

        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)

        classification = self.classifier(total)

        regression = self.con_layer(total)
        # print(self.total)
        return classification.squeeze(), regression.squeeze()