import re
import json
import random
from langdetect import detect
import language_tool_python
from transformers import pipeline

# Inisialisasi LanguageTool dan model analisis sentimen dengan model spesifik
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # Ganti dengan model yang Anda pilih
tokenizer_name = model_name  # Menggunakan tokenizer yang sama
tool = language_tool_python.LanguageTool('en-US')
classifier = pipeline('sentiment-analysis', model=model_name, tokenizer=tokenizer_name)  # Model dan tokenizer yang eksplisit

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = re.sub(r'[^\w\s\.,;:!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk memperbaiki tata bahasa dan ejaan
def correct_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Fungsi untuk menghapus duplikasi
def remove_duplicates(dataset):
    seen = set()
    unique_dataset = []
    for entry in dataset:
        content = entry["content"]
        if content not in seen:
            seen.add(content)
            unique_dataset.append(entry)
    return unique_dataset

# Fungsi untuk mendeteksi bahasa dan memastikan hanya bahasa Inggris
def filter_english(dataset):
    filtered_dataset = []
    for entry in dataset:
        content = entry["content"]
        try:
            if detect(content) == "en":
                filtered_dataset.append(entry)
        except:
            continue
    return filtered_dataset

# Fungsi untuk analisis sentimen dan menyaring konten positif
def is_positive_sentiment(text):
    result = classifier(text)
    return result[0]['label'] == 'POSITIVE'

# Fungsi untuk menghasilkan dataset dengan konten yang relevan dan nyambung
def generate_dataset(num_entries):
    dataset = []
    for _ in range(num_entries):
        entry_template = random.choice(template_data)  # Pilih template acak
        content = entry_template["content"]

        # Ganti placeholder dengan variabel acak
        content = content.replace("[project_name]", random.choice(project_names))
        content = content.replace("[technology]", random.choice(technologies))
        content = content.replace("[topic]", random.choice(topics))
        content = content.replace("[industry]", random.choice(industries))
        content = content.replace("[field]", random.choice(fields))
        content = content.replace("[issue]", random.choice(issues))
        content = content.replace("[action]", random.choice(actions))
        content = content.replace("[aspect]", random.choice(aspects))
        content = content.replace("[impact]", random.choice(impacts))
        content = content.replace("[benefit]", random.choice(benefits))
        content = content.replace("[related_technology]", random.choice(related_technologies))
        content = content.replace("[risk]", random.choice(risks))

        # Validasi: Pastikan jawaban tetap fokus dan sesuai dengan pertanyaan
        if "decentralization" in content:
            content = content.replace("[topic]", "decentralization")
            content = content.replace("[action]", "enhancing decentralized governance")

        # Memastikan bahwa jawaban tetap konsisten dan tidak ada frasa yang tidak relevan
        if "[issue]" in content:
            issue = random.choice(issues)
            content = content.replace("[issue]", issue)

        # Membersihkan dan memperbaiki tata bahasa
        content = clean_text(content)
        content = correct_grammar(content)

        # Hanya menyimpan konten yang memiliki sentimen positif
        if is_positive_sentiment(content):
            dataset.append({"content": content})

    return dataset

# Fungsi untuk menyimpan dataset ke file JSON
def save_dataset(dataset, filename):
    dataset = remove_duplicates(dataset)
    dataset = filter_english(dataset)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"Dataset saved to {filename}.")

# Definisi Template dan Variabel (sudah Anda berikan)
template_data = [
    {
        "content": "Question: What is the significance of Nakamoto's original whitepaper in the development of blockchain technology? Answer: Nakamoto's whitepaper laid the foundation for decentralized systems by introducing the concept of a trustless, peer-to-peer network, which solved the double-spending problem and created a decentralized financial ecosystem."
    },
    {
        "content": "Question: How does [project_name] implement Nakamoto's principle of decentralization in its architecture? Answer: [project_name] uses a distributed ledger system and consensus mechanisms to ensure that no single entity controls the network, embodying Nakamoto's principle of decentralization in its architecture."
    },
    {
        "content": "Question: How does Nakamoto's vision for decentralized governance influence the development of [project_name]? Answer: Nakamoto's vision of decentralized governance inspired [project_name] to create a system where decisions are made by consensus rather than centralized authorities, enhancing the autonomy of its users."
    },
    {
        "content": "Question: How does the introduction of [technology] in [project_name] contribute to Nakamoto's goal of secure, scalable, and private decentralized finance (DeFi)? Answer: The integration of [technology] in [project_name] improves transaction security, enhances scalability, and preserves user privacy, aligning with Nakamoto's vision for a secure and scalable DeFi ecosystem."
    },
    {
        "content": "Question: How does Nakamoto's belief in the importance of cryptographic integrity shape the development of [project_name]? Answer: Nakamoto's belief in cryptographic integrity is reflected in [project_name] through its use of [technology], which ensures data security and user trust in a decentralized environment."
    },
    {
        "content": "Question: What are the key ways in which [project_name] addresses scalability challenges, a key issue highlighted by Nakamoto in the Bitcoin whitepaper? Answer: [project_name] addresses scalability through [action], which allows the system to handle an increasing number of transactions while maintaining decentralization and security."
    },
    {
        "content": "Question: In what ways does [project_name] ensure user privacy while maintaining decentralization, a balance central to Nakamoto's vision? Answer: [project_name] uses [technology] to protect user privacy without sacrificing the decentralized nature of the network, staying true to Nakamoto's principles."
    },
    {
        "content": "Question: How does Nakamoto's focus on economic incentives in the Bitcoin protocol relate to [project_name]'s tokenomics? Answer: Nakamoto's focus on incentivizing participants through a proof-of-work system is mirrored in [project_name] through its tokenomics, where users are rewarded for their contributions to network security and growth."
    },
    {
        "content": "Question: How does [project_name] balance the need for scalability with Nakamoto's emphasis on decentralization and security in the blockchain space? Answer: [project_name] achieves this balance through [action], ensuring that scalability improvements do not compromise the decentralized nature or security of the network."
    },
    {
        "content": "Question: How does the implementation of [technology] in [project_name] reflect Nakamoto's commitment to creating a system with immutable records? Answer: [project_name] uses [technology] to ensure that all records are permanent and unchangeable, aligning with Nakamoto's commitment to immutability in blockchain technology."
    }
]

project_names = ["Tribus", "Novus", "BlockchainX", "Web3Wallet", "CryptoEdge", "BlockSphere", "SatoshiNode", "BlockFusion", "DecentraX", "ProofTech"]
technologies = ["cryptographic hashing", "zero-knowledge proofs", "layer-2 scaling solutions", "decentralized exchanges", "privacy-preserving encryption", "interoperable protocols", "privacy-focused smart contracts", "proof-of-stake", "proof-of-work", "lightning network"]
topics = ["decentralization", "blockchain security", "cryptocurrency scalability", "peer-to-peer networking", "privacy preservation", "self-sovereign identity", "trustless systems", "immutable ledger", "cryptographic integrity", "decentralized governance"]
industries = ["finance", "healthcare", "logistics", "technology", "energy", "entertainment", "gaming", "digital art", "supply chain", "IoT"]
fields = ["blockchain technology", "decentralized finance", "cryptocurrency", "distributed ledger technology", "Web3", "smart contract platforms", "tokenomics", "protocol governance", "cross-chain interoperability"]
issues = ["centralization risks", "privacy concerns", "scalability challenges", "transaction delays", "network congestion", "regulatory compliance", "energy consumption", "market manipulation", "data integrity"]
actions = ["optimizing smart contracts", "enhancing transaction speed", "securing data integrity", "scaling decentralized systems", "improving user experience", "fostering ecosystem collaboration", "creating interoperable solutions", "minimizing energy usage", "building robust consensus algorithms"]
aspects = ["security", "scalability", "privacy", "decentralization", "governance", "interoperability", "efficiency", "trust", "inclusivity", "sustainability"]
impacts = ["reduced fraud", "increased transparency", "enhanced privacy", "improved efficiency", "greater financial inclusion", "lower costs", "faster transactions", "empowered communities", "sustainable growth", "global access"]
benefits = ["lower transaction fees", "improved liquidity", "enhanced user control", "secure digital identity", "fast and efficient cross-border payments", "increased scalability", "privacy-focused features", "transparent governance", "decentralized data storage", "self-sovereign assets"]
related_technologies = ["AI", "IoT", "5G", "quantum computing", "cloud computing", "edge computing", "cybersecurity", "tokenization", "NFTs", "metaverse"]
risks = ["smart contract bugs", "network attacks", "regulatory uncertainty", "liquidity risks", "market volatility", "privacy breaches", "technical failures", "centralized exchanges", "adoption barriers", "forks in protocols"]

# Fungsi utama untuk menghasilkan dan menyimpan dataset
dataset = generate_dataset(10000)  # Generate 100 entries for the dataset
save_dataset(dataset, "dataset-v6.json")
