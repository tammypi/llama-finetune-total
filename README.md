# 在AutoDL平台上使用RTX4090 24G显卡微调LLama 7B

## 1.算力租用平台

AutoDL:https://www.autodl.com/

GPU型号：RTX 4090 24G

费用计算：

无卡开机：0.1元/小时

带卡开机：2.15元/小时

## 2.无卡开机配置环境

### 2.1.下载LLama源码

```
git clone https://github.com/facebookresearch/llama.git
```

### 2.2.下载7B模型

```
cd /root/autodl-tmp/llama
mkdir models
cd models
mkdir 7B
wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model -O ./tokenizer.model
wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk -O ./tokenizer_checklist.chk
wget https://agi.gpt4.org/llama/LLaMA/7B/consolidated.00.pth -O ./7B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/7B/params.json -O ./7B/params.json
wget https://agi.gpt4.org/llama/LLaMA/7B/checklist.chk -O ./7B/checklist.chk
```

### 2.3.下载微调代码

```
cd /root/autodl-tmp
git clone https://github.com/chaoyi-wu/Finetune_LLAMA.git
```

用https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py替换Finetune_LLAMA的LLAMA_Model/convert_llama_weights_to_hf.py

### 2.4 下载Lit-LLama代码

```
git clone https://github.com/Lightning-AI/lit-llama.git
```

## 3.启动微调

此时需要带卡开机。

### 3.1.转换模型原始权重

将模型的原始权重，转换为PyTorch的格式。

```
cd /root/autodl-tmp/Finetune_LLAMA/LLAMA_Model
python convert_llama_weights_to_hf.py --input_dir /root/autodl-tmp/llama/models --model_size 7B --output_dir /root/autodl-tmp/models
cd /root/autodl-tmp/lit-lamma
python scripts/convert_hf_checkpoint.py --checkpoint_dir /root/autodl-tmp/models --model_size 7B
```

执行完命令后：

/root/autodl-tmp/models路径下会有转换后的模型

/root/autodl-tmp/lit-llama/checkpoints/lit-llama/7B下会有lit-llama.pth

### 3.2.下载训练数据并处理

mkdir /root/autodl-tmp/trainData

从链接：https://drive.google.com/file/d/1OqfOUWYfrK6riE9erOx-Izp3nItfqz_K/view 下载instinwild_ch.json

执行命令：

```
cd /root/autodl-tmp/lit-lamma
python scripts/prepare_alpaca.py
```

训练数据路径/root/autodl-tmp/trainData下会出现train.pt和test.pt

说明：

这是个中文问答的训练数据，因为LLama 7B模型默认是不支持中文的，所以用中文问答来微调，使得模型具备回复中文问题的能力。

该训练数据文本摘录2条如下：

```
{
    "instruction": "请讲解如何缓解上班族病的症状。",
    "input": "",
    "output": "一种有效的缓解上班族病的症状方法是出去散步，每天晚上可以花几个小时去散步，减少坐姿固定的时间，放松肩痛、腰痛和背部综合症的发作。另外，可以试着利用午休时间或者其他空余时间锻炼一下，比如慢跑，打太极拳等，帮助舒缓，运动释放时也可以练习深呼吸，这能帮助消除压力，更有利于解除病症。",
    "id": 0
},
{
    "instruction": "描述一下优质的客户关系管理模式。",
    "input": "",
    "output": "优质的客户关系管理模式是指，通过建立客户之间的关系，建立并提供优质的服务，将客户的满意度转变为重复购买概念。这种模式强调人性化服务，以及知识和技能的结合，建立关系是一种长期的过程，而且可以建立起客户的忠诚度和口碑好评。该模式还可以培养客户之间的信任关系，增强客户感受到优质服务的同时，建立起长期客户风险防范机制，以及客户满意度机制，使企业拥有稳定、可持续和良好的客户关系管理能力。",
    "id": 1
}
```



### 3.3 启动微调

修改/root/autodl-tmp/lit-llama/lora.py

```
def main(
    data_dir: str = "/root/autodl-tmp/trainData", 
    pretrained_path: str = "/root/autodl-tmp/lit-llama/checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "/root/autodl-tmp/models/tokenizer.model",
    out_dir: str = "/root/autodl-tmp/out",
):
```

注意这几个参数，默认值设置为当前环境上的路径。

然后执行微调：

```
cd /root/autodl-tmp/lit-llama
python finetune/lora.py
```

微调大概耗费了3个小时。

### 3.4 测试微调之后的结果

修改/root/autodl-tmp/lit-llama/generate/lora.py

```
def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    #微调后的权重
    lora_path: Path = Path("/root/autodl-tmp/out/lit-llama-lora-finetuned.pth"),
    #之前的7B权重和模型
    pretrained_path: Path = Path("/root/autodl-tmp/lit-llama/checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("/root/autodl-tmp/models/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
```

主要修改lora_path、pretrained_path、tokenizer_path三个路径的值，修改为自己环境的值。

运行效果如下：

![image-20230814173542996]([.\image\1.png](https://github.com/tammypi/llama-finetune-total/blob/main/image/1.png?raw=true))

![image-20230814173729758]([.\image\2.png](https://github.com/tammypi/llama-finetune-total/blob/main/image/2.png?raw=true>))

可以回答中文问题了，只是回答存在中断的现象，也存在部分胡言乱语。待进一步研究。

微调前后的效果对比：

![3]([.\image\3.png](https://github.com/tammypi/llama-finetune-total/blob/main/image/3.png?raw=true)
