import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms, models

class CNN_ChangiAeroVisionModel(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super(CNN_ChangiAeroVisionModel, self).__init__()
        self.backbone = models.resnet50(weights=None) 
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

class IntentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True) 
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

st.set_page_config(page_title="Changi Operations AI", layout="wide")
device = torch.device('cpu') 

@st.cache_resource
def load_resources():
    resources = {}
    
    # INTENT
    try:
        resources['intent_vocab'] = pickle.load(open("models/intent_vocab.pkl", "rb"))
        resources['intent_labels'] = pickle.load(open("models/intent_labels.pkl", "rb"))
        resources['intent_model'] = IntentLSTM(len(resources['intent_vocab']), 64, 128, len(resources['intent_labels']))
        resources['intent_model'].load_state_dict(torch.load("models/intent_model.pth", map_location=device))
        resources['intent_model'].eval()
    except: pass

    # SENTIMENT
    try:
        resources['sent_vocab'] = pickle.load(open("models/sentiment_vocab.pkl", "rb"))
        # Sentiment usually has 3 classes (Neg, Neu, Pos)
        resources['sent_model'] = SentimentRNN(len(resources['sent_vocab']), 64, 128, 3) 
        resources['sent_model'].load_state_dict(torch.load("models/sentiment_model.pth", map_location=device))
        resources['sent_model'].eval()
    except: pass

    # CNN
    try:
        resources['cnn_model'] = CNN_ChangiAeroVisionModel(num_classes=8) 
        checkpoint = torch.load("models/aircraft_model.pth", map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            resources['cnn_model'].load_state_dict(checkpoint['model_state_dict'])
        else:
            resources['cnn_model'].load_state_dict(checkpoint)
        resources['cnn_model'].eval()
    except Exception as e:
        resources['cnn_error'] = str(e)

    return resources

res = load_resources()

def preprocess_text_debug(text, vocab, max_len=20):
    tokens = text.lower().split()
    unk_id = vocab.get('<UNK>', 1)
    indices = []
    debug_tokens = []
    
    for t in tokens:
        if t in vocab:
            indices.append(vocab[t])
            debug_tokens.append(t)
        else:
            indices.append(unk_id)
            debug_tokens.append(f"{t}(UNK)")
            
    if len(indices) < max_len: indices += [0] * (max_len - len(indices))
    else: indices = indices[:max_len]
        
    return torch.LongTensor([indices]), debug_tokens

st.title("✈️ Changi AeroVision & Ops AI")
st.caption("Terminal 5 Automated Command Center")

tab1, tab2 = st.tabs(["💬 Ops Command (NLP)", "📷 Visual Scan (CNN)"])

# NLP
with tab1:
    col_input, col_view = st.columns([1, 1])
    with col_input:
        user_text = st.text_area("Passenger Query:", "My flight is delayed and I am very angry!")
        run_nlp = st.button("Analyze Query", type="primary")

    if run_nlp:
        # INTENT ANALYSIS
        st.divider()
        st.markdown("### 1. Intent Analysis (Eden)")
        col_i1, col_i2 = st.columns(2)
        
        if 'intent_model' in res:
            with torch.no_grad():
                tensor_in, debug_toks = preprocess_text_debug(user_text, res['intent_vocab'])
                outputs = res['intent_model'](tensor_in)
                probs = F.softmax(outputs, 1)
                
                # Top 3
                top3_prob, top3_idx = torch.topk(probs, 3)
                labels = res['intent_labels']
                results = []
                for i in range(3):
                    idx = top3_idx[0][i].item()
                    name = labels[idx] if idx < len(labels) else f"ID {idx}"
                    results.append({"Intent": name, "Confidence": top3_prob[0][i].item()})
                
                with col_i1:
                    st.info(f"**Detected:** {results[0]['Intent'].upper()}")
                    with st.expander("Show Tokenizer Debug"):
                        st.code(debug_toks)
                with col_i2:
                    st.bar_chart(pd.DataFrame(results).set_index("Intent"))
        else:
            st.warning("Intent Model Not Loaded")

        # SENTIMENT ANALYSIS
        st.markdown("### 2. Sentiment Analysis (Chase)")
        col_s1, col_s2 = st.columns(2)
        
        if 'sent_model' in res:
            with torch.no_grad():
                tensor_in, debug_toks_s = preprocess_text_debug(user_text, res['sent_vocab'])
                outputs = res['sent_model'](tensor_in)
                probs = F.softmax(outputs, 1)
                
                # Classes: 0=Neg, 1=Neu, 2=Pos
                sent_labels = ["Negative", "Neutral", "Positive"]
                
                # Create Dataframe for Chart
                sent_data = []
                for i, label in enumerate(sent_labels):
                    sent_data.append({"Tone": label, "Score": probs[0][i].item()})
                
                best_idx = torch.argmax(probs).item()
                best_label = sent_labels[best_idx]
                
                with col_s1:
                    if best_label == "Negative": st.error(f"**Tone:** {best_label}")
                    elif best_label == "Positive": st.success(f"**Tone:** {best_label}")
                    else: st.warning(f"**Tone:** {best_label}")
                    
                    with st.expander("Show Sentiment Debug"):
                        st.code(debug_toks_s)
                        st.write("Check if emotional words like 'angry' become (UNK)")

                with col_s2:
                    st.bar_chart(pd.DataFrame(sent_data).set_index("Tone"))
        else:
            st.warning("Sentiment Model Not Loaded (Check models/ folder)")

# CNN
with tab2:
    st.subheader("Aircraft Identification")
    
    # SIDEBAR CONTROLS
    st.sidebar.header("CNN Settings")
    use_norm = st.sidebar.checkbox("Use ImageNet Normalization", value=True, 
                                   help="Uncheck this if accuracy is bad! Some models prefer Raw images.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    
    if uploaded_file and 'cnn_model' in res:
        image = Image.open(uploaded_file).convert('RGB')
        col_img, col_cnn_stats = st.columns(2)
        
        with col_img:
            st.image(image, caption='Gate Feed', use_column_width=True)
        
        with col_cnn_stats:
            if st.button("Identify Aircraft"):
                with st.spinner("Analyzing..."):
                    with torch.no_grad():
                        # DYNAMIC PREPROCESSING
                        if use_norm:
                            # Standard ImageNet Stats
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                        else:
                            # Raw Resize
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])
                            
                        img_tensor = transform(image).unsqueeze(0)
                        
                        outputs = res['cnn_model'](img_tensor)
                        probs = F.softmax(outputs, dim=1)
                        
                        top3_prob, top3_idx = torch.topk(probs, 3)
                        
                        # CLASS NAMES (UPDATE THIS LIST!)
                        # Based on typical datasets, these are the 8 most likely classes
                        classes = [
                            'A320', 'A380', 'B737', 'B747', 
                            'B777', 'B787', 'CRJ-900', 'ERJ-190'
                        ]
                        
                        results = []
                        for i in range(3):
                            idx = top3_idx[0][i].item()
                            name = classes[idx] if idx < len(classes) else f"Class {idx}"
                            results.append({"Type": name, "Prob": top3_prob[0][i].item()})
                        
                        best = results[0]
                        st.success(f"**Identified:** {best['Type']}")
                        st.caption(f"Confidence: {best['Prob']*100:.1f}%")
                        
                        st.bar_chart(pd.DataFrame(results).set_index("Type"))