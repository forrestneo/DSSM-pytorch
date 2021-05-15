
import pandas as pd
import numpy as np
import datetime
import time
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import faiss
#from dnn_model import DNN
#from encoder_model import Encoder
#from weight_initiallizer import Initializer
#from itemcf import itemcf_sim

# 数据预处理
root_path = os.path.abspath('.')
user = pd.read_csv(os.path.join(root_path, 'ml-1m', 'users.dat'), sep='::', names = ['user', 'gender', 'age', 'occupation', 'zip_code'],engine='python')
movie = pd.read_csv(os.path.join(root_path, 'ml-1m', 'movies.dat'), sep='::', names = ['movie', 'title', 'genres'],engine='python')
rating = pd.read_csv(os.path.join(root_path, 'ml-1m', 'ratings.dat'), sep='::', names = ['user', 'movie', 'ratings', 'timestamp'],engine='python')
# mapping
user_id_dict = dict()
for idx, uid in enumerate(user['user'].tolist()):
    user_id_dict[uid] = idx
movie_id_dict = dict()
for idx, mid in enumerate(movie['movie'].tolist()):
    movie_id_dict[mid] = idx
user['user'] = user['user'].map(user_id_dict)
movie['movie'] = movie['movie'].map(movie_id_dict)
rating['user'] = rating['user'].map(user_id_dict)
rating['movie'] = rating['movie'].map(movie_id_dict)

​
​
# 时间处理
rating['timestamp'] = rating['timestamp'].apply(lambda x: time.localtime(x))
rating['time_str'] = rating['timestamp'].apply(lambda x: 
                                                  time.strftime("%Y-%m-%d %H:%M:%S",x))
​
# 切分训练和验证集，数据是2000到2003年的，以2003年用户的行为为验证集，用用户2000-2002年数据进行预测和验证，过滤掉那些只在2003年有行为的用户
train_data = rating[rating['time_str']<'2003-01-01 00:00:00']
val_data = rating[rating['time_str']>='2003-01-01 00:00:00']
# 过滤
val_data = val_data[val_data['user'].isin(train_data['user'].unique())]

# DSSM loss 模型
# 随机负采样
sample_list = list(train_data['movie'].unique())
data = list()
for idx, rows in tqdm(train_data.iterrows(), total=len(train_data)):
    use = rows['user']
    mov = rows['movie']
    data.append([use, mov, 1])
    for m in np.random.choice(sample_list, 3):
        data.append([use, m, 0])
data = pd.DataFrame(data, columns=['user', 'movie', 'tag'])
data.head()

# 合并用户特征和电影特征
le = LabelEncoder()
user['gender'] = le.fit_transform(user['gender'])
user['age'] = le.fit_transform(user['age'])
user['occupation'] = le.fit_transform(user['occupation'])
​
data = pd.merge(data, user[['user', 'gender', 'age', 'occupation']], how='left', on='user')
​
genres = list()
tmp = movie['genres'].apply(lambda x: x.split('|'))
for l in tmp.tolist():
    genres += l
    
genres_dict = dict()
for idx, g in  enumerate(list(set(genres))):
    genres_dict[g] = idx + 1
    
movie['genres'] = tmp.apply(lambda x: [genres_dict[i] for i in x])
​
data = pd.merge(data, movie[['movie', 'genres']], how='left', on='movie')


data.head()

# 定义训练网络
class trainset(Dataset):
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
​
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        data = (x, y)
        return data
​
    def __len__(self):
        return len(self.x)
​
# 训练集验证集随机分割
train_df, test_df = train_test_split(data, test_size=0.2, random_state=2021)
train_x = train_df[['user', 'gender', 'age', 'occupation', 'movie']].values
train_y = train_df['tag'].values
test_x = test_df[['user', 'gender', 'age', 'occupation', 'movie']].values
test_y = test_df['tag'].values
​
# 构造dataloader
train_dataset = trainset((train_x, train_y))
test_dataset = trainset((test_x, test_y))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# DSSM MODEL
class DNNModel(nn.Module):
    def __init__(self, inp_user, inp_movie, out, input_user_categorical_feature, input_movie_categorical_feature, 
                 hidden_layers, dropouts, batch_norm):
        super(DNNModel, self).__init__()
        self.user_embed = nn.Embedding(input_user_categorical_feature[0][0], input_user_categorical_feature[0][1])
        self.gender_embed = nn.Embedding(input_user_categorical_feature[1][0], input_user_categorical_feature[1][1])
        self.age_embed = nn.Embedding(input_user_categorical_feature[2][0], input_user_categorical_feature[2][1])
        self.occupation_embed = nn.Embedding(input_user_categorical_feature[3][0], input_user_categorical_feature[3][1])
        self.movie_embed = nn.Embedding(input_movie_categorical_feature[0][0], input_movie_categorical_feature[0][1])
#         self.genres_embed = nn.Embedding(input_movie_categorical_feature[1][0], input_movie_categorical_feature[1][1])
        
        self.user_dnn = nn.Sequential(
            nn.Linear(512, 128),
#             nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
        
        self.movie_dnn = nn.Sequential(
            nn.Linear(128, 128),
#             nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
​
        
    def forward(self, x):
        u = self.user_embed(x[:, 0])
        g = self.gender_embed(x[:, 1])
        a = self.age_embed(x[:, 2])
        oc = self.occupation_embed(x[:, 3])
        m = self.movie_embed(x[:, 4])
        
        u = torch.cat([u, g, a, oc], -1)
        u = self.user_dnn(u)
        m = self.movie_dnn(m)
        u = u/torch.sum(u*u, 1).view(-1,1)
        m = m/torch.sum(m*m, 1).view(-1,1)
        return u, m

# train_model
def train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, path, early_stop):
    """
    pytorch 模型训练通用代码
    :param model: pytorch 模型
    :param train_loader: dataloader, 训练数据
    :param val_loader: dataloader, 验证数据
    :param epoch: int, 训练迭代次数
    :param loss_function: 优化损失函数
    :param optimizer: pytorch优化器
    :param path: save path
    :param early_stop: int, 提前停止步数
    :return: None
    """
    # 是否使用GPU
  #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    
    # 训练
    for i in range(epoch):
        total_loss, count = 0, 0
        y_pred = list()
        y_true = list()
        for idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y = x.to(device), y.to(device) 
            u, m = model(x)
            predict = torch.sigmoid(torch.sum(u*m, 1))
            y_pred.extend(predict.cpu().detach().numpy())
            y_true.extend(y.cpu().detach().numpy())
            loss = loss_function(predict, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
            
        train_auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        torch.save(model, path.format(i+1))
        print("Epoch %d train loss is %.3f and train auc is %.3f" % (i+1, total_loss / count, train_auc))
    
        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        val_y_pred = list()
        val_true = list()
        for idx, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y = x.to(device), y.to(device)
            u, m = model(x)
            predict = torch.sigmoid(torch.sum(u*m, 1))
            val_y_pred.extend(predict.cpu().detach().numpy())
            val_true.extend(y.cpu().detach().numpy())
            loss = loss_function(predict, y.float())
            total_eval_loss += float(loss)
            count_eval += 1
        val_auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        print("Epoch %d val loss is %.3fand train auc is %.3f" % (i+1, total_eval_loss / count_eval, val_auc))
        
        # 提前停止策略
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break

# 模型初始化
inp_user = 128
inp_movie = 128
out = 64
input_user_categorical_feature = {0: (6040, 128), 1: (2, 128), 2: (7, 128), 3: (21, 128)}
input_movie_categorical_feature =  {0: (3883, 128), 1:(18, 128)}
hidden_layers = [128, 64]
dropouts = [0.5, 0.5, 0.5]
batch_norm = False
​
model = DNNModel(inp_user, inp_movie, out, input_user_categorical_feature, input_movie_categorical_feature, 
                 hidden_layers, dropouts, batch_norm)
​
# 模型训练
epoch = 20
loss_function = F.binary_cross_entropy_with_logits
early_stop = 3
learn_rate = 0.004
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
path = 'model/model_{}.pth'
​
train_model(model, train_loader, test_loader, epoch, loss_function, optimizer, path, early_stop)

# 结果验证
model.eval()
user['movie'] = 1
test_x = user[['user', 'gender', 'age', 'occupation', 'movie']].values
x = torch.from_numpy(test_x).cuda()
user_embed, _ = model(x)
​
movie['user'] = 1
movie['gender'] = 1
movie['age'] = 1
movie['occupation'] = 1
test_x = movie[['user', 'gender', 'age', 'occupation','movie']].values
x = torch.from_numpy(test_x).cuda()
_, movie_embed = model(x)
​
movie_embed = movie_embed.cpu().detach().numpy()
user_embed = user_embed.cpu().detach().numpy()
​
​
# faiss索引构建
d = 64
nlist = 10
index = faiss.IndexFlatL2(d)
index.add(movie_embed)
​
# 验证集数据字典化
user_movie_dict_val = dict()
for idx, rows in tqdm(val_data.iterrows(), total=len(val_data)):
    u = rows['user']
    m = rows['movie']
    if u not in user_movie_dict_val:
        user_movie_dict_val[u] = [m]
    else:
         user_movie_dict_val[u].append(m)
            
# 用户推荐结果索引           
D, I = index.search(user_embed[list(val_data['user'].unique())], 50)
​
# 召回率计算
hits, total = 0, 0
for uid, rec_list in zip(list(val_data['user'].unique()), I):
    hits += len(set(rec_list)&set(user_movie_dict_val[uid]))
    total += len(user_movie_dict_val[uid])
print("recall is %.3f" % (hits/total))

