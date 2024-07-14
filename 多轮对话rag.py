import openai
from sentence_transformers import SentenceTransformer, util

# ChatGPT-4 API 认证
openai.api_key = 'your-api-key'

# 初始化嵌入模型
# 更多模型相关内容请见https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 假设知识库中的文档
knowledge_docs = [
    "Diversify your portfolio to spread risk.",
    "Do thorough research before investing in any stock.",
    "Monitor the market trends and adjust your investments accordingly."
]

# 预计算知识库文档的嵌入
knowledge_embeddings = embedder.encode(knowledge_docs, convert_to_tensor=True)

def retrieve_knowledge(user_input):
    # 生成用户输入的嵌入
    user_input_embedding = embedder.encode(user_input, convert_to_tensor=True)
    
    # 计算相似度并检索最相关的文档
    similarities = util.pytorch_cos_sim(user_input_embedding, knowledge_embeddings)
    most_similar_idx = similarities.argmax()
    
    return knowledge_docs[most_similar_idx]

def generate_response(user_id, conversation_history, user_input):
    # 更新对话历史
    conversation_history.append({"role": "user", "content": user_input})
    
    # 拼接对话历史用于上下文
    context = ' '.join([entry['content'] for entry in conversation_history])
    
    # 检索相关知识
    knowledge = retrieve_knowledge(user_input)
    
    # 包含用户上下文信息和检索知识的请求
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": f"User ID: {user_id}"},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": knowledge}
        ]
    )
    
    # 更新对话历史
    conversation_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    
    return response['choices'][0]['message']['content'], conversation_history

# 示例调用
user_id = 'feixiaomao'
conversation_history = []

# 第一次用户输入
user_input_1 = 'Can you give me some tips on investing in stocks?'
response_1, conversation_history = generate_response(user_id, conversation_history, user_input_1)
print(f"User: {user_input_1}")
print(f"Assistant: {response_1}")

# 第二次用户输入
user_input_2 = 'What are the risks involved?'
response_2, conversation_history = generate_response(user_id, conversation_history, user_input_2)
print(f"User: {user_input_2}")
print(f"Assistant: {response_2}")

# 第三次用户输入
user_input_3 = 'Can you suggest some good stocks to invest in?'
response_3, conversation_history = generate_response(user_id, conversation_history, user_input_3)
print(f"User: {user_input_3}")
print(f"Assistant: {response_3}")
