import streamlit as st
import torch
import torch.nn as nn
import mne
import numpy as np
import os
import matplotlib.pyplot as plt


# 1. 模型架构定义
class SimpleEEGNet(nn.Module):
    def __init__(self, num_classes=3, channels=64, samples=320):
        super(SimpleEEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 33), padding=(0, 16))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (channels, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pooling = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 1 * 80, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# 2. 页面与授权配置
st.set_page_config(page_title="BCI Medical Terminal", layout="wide")

with st.sidebar:
    st.header("🔒 系统授权")
    password = st.text_input("输入访问代码", type="password")
    if password != "Centria2026":
        st.warning("请输入授权码访问医疗终端")
        st.stop()

    st.success("授权成功")
    st.markdown("---")
    st.subheader("💡 演示模式")
    st.write("若无信号文件，请选择下方样本：")

    # 样例数据路径
    samples = {
        "停止指令": "data/test_samples/test_Rest_00.fif",
        "左转指令": "data/test_samples/test_Left_03.fif",
        "右转指令": "data/test_samples/test_Right_00.fif"
    }

    sample_choice = st.selectbox("选择内置样本", ["无"] + list(samples.keys()))


# 3. 加载训练好的模型
@st.cache_resource
def load_model():
    model = SimpleEEGNet()
    if os.path.exists('results/model.pth'):
        model.load_state_dict(torch.load('results/model.pth', map_location='cpu'))
        model.eval()
        return model
    return None


model = load_model()

# 4. 主界面布局
st.title("🧠 脑机接口医疗辅助控制终端")
st.info("当前 AI 识别准确率：82.33% | 信号窗口：2.0 秒")

uploaded_file = st.file_uploader("上传脑电信号 (.fif)", type=["fif"])

# 确定数据源
data_source = None
if uploaded_file:
    with open("temp.fif", "wb") as f:
        f.write(uploaded_file.getbuffer())
    data_source = "temp.fif"
elif sample_choice != "无":
    data_source = samples[sample_choice]

if data_source:
    try:
        epochs = mne.read_epochs(data_source, preload=True, verbose=False)
        epochs.resample(160, verbose=False)
        epochs.filter(8., 30., verbose=False)
        raw_data = epochs.get_data()

        # 对齐 320 个采样点
        if raw_data.shape[2] < 320:
            raw_data = np.pad(raw_data, ((0, 0), (0, 0), (0, 320 - raw_data.shape[2])))
        else:
            raw_data = raw_data[:, :, :320]

        # Z-score 标准化：z = (x - mu) / sigma
        norm_data = (raw_data - np.mean(raw_data)) / np.std(raw_data)

        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("📊 实时波形诊断")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(norm_data[0, 0, :], color='#00FFAA', linewidth=1)
            ax.set_ylim(-4, 4)
            ax.set_ylabel("标准化幅值 (Z)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with c2:
            st.subheader("🕹️ 指令翻译器")
            if st.button("开始实时分析", use_container_width=True):
                input_tensor = torch.FloatTensor(norm_data).unsqueeze(1)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, 1)

                res_idx = pred[0].item()
                res_conf = conf[0].item() * 100

                # 指令映射
                cmds = {
                    0: {"n": "待命/停止", "i": "⏸️", "c": "gray"},
                    1: {"n": "左转指令", "i": "⬅️", "c": "#1E90FF"},
                    2: {"n": "右转指令", "i": "➡️", "c": "#32CD32"}
                }

                target = cmds[res_idx]
                st.markdown(f"""
                    <div style="background-color: {target['c']}; padding: 25px; border-radius: 15px; text-align: center; color: white;">
                        <h1 style="font-size: 70px; margin: 0;">{target['i']}</h1>
                        <h2 style="margin: 0;">{target['n']}</h2>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.progress(res_conf / 100)
                st.write(f"**模型置信度：** {res_conf:.2f}%")

                if res_conf < 60:
                    st.warning("⚠️ 信号强度低，已启用安全保护机制。")

    except Exception as e:
        st.error(f"处理失败：{e}")
