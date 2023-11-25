# Classification_of_Pediatric_Psychiatric_Disorders

## Reference
These data are collected in National Taiwan University Hospital Hsin-Chu Branch.  
For the detail of this experiment, please read my master thesis.

## About
- Collaborated with the Pediatrics Department of NTUH Hsinchu Branch.
- Utilized near-infrared spectroscopy to measure prefrontal cortex oxygenation changes in pediatric patients with common mental disorders during testing.
- Constructed a CNN model for multi-class classification, distinguishing between healthy and diseased children. Achieved an accuracy rate of approximately 80%.
- Applied explainable AI to visualize the results.

## Files
```
 Path
    ├ 1_data_preprocessing.ipynb
    └ 2_deep_learning.ipynb
```

## Data Preprocessing
Four steps is operated in this file
1. Caculating oxygen concentration of blood
2. Data cleaning
3. Filtering
4. Normalization

## Deep Learning
1. Preprocessing:
 - Load data and add labels.
 - Pad the data to same size.
```python
max_len = 0
for file in all_df['data']:
    data = pd.read_csv(file)
    if len(data) > max_len: 
        max_len = len(data)
print(max_len)
blank = pd.Series(range(max_len))
for file in all_df['data']:
    data = pd.read_csv(file, usecols=[i for i in range(3, 7)])
    data_pad = pd.concat([blank, data], axis=1)
    data_pad = data_pad.fillna(0).drop(columns=[0])
    path = file.replace('input', 'working')
    if not os.path.isdir(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    
    avg = []
    for i in range(int(max_len/17)):
        avg.append(data_pad.iloc[i*17:(i+1)*17].mean(axis=0).values)
    
    data_final = pd.DataFrame(avg)
    data_final.T.to_csv(path, header=False, index=False)
```
2. Define dataset:
```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.df = annotations_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = pd.read_csv(img_path, header=None).values
        name = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, name, label
```
3. Define net:
```python
class Network(nn.Module):
    def __init__(self, pool = 6, fc1=1024, maxpool=2, conv8_open=False):
        super(Network, self).__init__()
        
        self.maxpool = maxpool
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=10, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)        
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(self.maxpool, padding= 1)
        
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(64)   
        self.pool2 = nn.MaxPool1d(self.maxpool, padding= 1)

        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=10, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1, padding=1)
        self.bn8 = nn.BatchNorm1d(128)
        
        self.conv_open = conv8_open
        self.pools = pool
        self.ave_pool = nn.AdaptiveAvgPool1d(self.pools)
        
        # FC
        self.fc1_num = fc1
        self.fc1 = nn.Linear(128*self.pools, self.fc1_num)
        self.fc2 = nn.Linear(self.fc1_num, 64)
        self.fc3 = nn.Linear(64, 3)

        self.soft = nn.Softmax(dim=1)

        self.drop = nn.Dropout(0.1)

    def forward(self, input1):
        output = F.celu(self.conv1(input1))
        output = F.celu(self.conv2(output))
        output = self.bn2(output)
        output = self.pool(output) 
        
        output = F.celu(self.conv4(output))     
        output = F.celu(self.conv5(output)) 
        output = self.bn5(output)
        output = self.pool(output)   
        output =  F.celu(self.conv6(output))
        output =  F.celu(self.conv7(output))

        if self.conv_open:
            output =  F.elu(self.conv8(output))
        
        output = self.ave_pool(output)
        
        output = output.view(-1, 128*self.pools) 
        
        con = self.fc1(output)
        con = self.drop(con)
        con = self.fc2(con)   
        con = self.fc3(con)

        con = self.soft(con)

        return con
```
4. Start training:
 - Use `StratifiedKFold` to split data.
 - Define evalution and training function.
 - Traing the model and record the accuracies.

5. Reproduction:
 - Load best model
 - Use `confusion_matrix` to evaluate the predictions.

6. Explainable AI -- Intergradient Gradient:
 - Use `captum` module.
```python
model.eval()
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input1, target=0, return_convergence_delta=True)

df_cam = pd.DataFrame()
for nums, att in enumerate(attributions):
    df_cam[f'{nums}_cam0'] = att.cpu()[0]
    df_cam[f'{nums}_cam1'] = att.cpu()[1]
    df_cam[f'{nums}_cam2'] = att.cpu()[2]
    df_cam[f'{nums}_cam3'] = att.cpu()[3]
```
 - Overlay the attributions with original data
```python
 yy = np.linspace(-0.01, 1, 100)
 cmapp1 = [(1, 0, 0, a*0.5) for a in df_rolling[f'{nums}_cam0']]
 colors = cmapp1
 # 繪製水平色彩漸進圖
 fig, ax = plt.subplots(figsize=(40, 15))

 for i in range(len(colors)):
     ax.plot(np.ones_like(yy) * i, yy, color=colors[i], linewidth=4)
     
 plt.scatter(np.arange(0, len(df_rolling[f'{nums}_cam0']), 1), df_rolling[f'{nums}_cam0'], label= 'region1', c='black', linewidth=10)
 plt.show()
```
