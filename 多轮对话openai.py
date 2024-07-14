import openai
import numpy as np

# 设置 OpenAI API 密钥
openai.api_key = 'your-api-key'

# 假设知识库中的文档
knowledge_docs = [
    "Diversify your portfolio to spread risk.",
    "Do thorough research before investing in any stock.",
    "Monitor the market trends and adjust your investments accordingly."
]

# 生成知识库文档的嵌入向量
def get_embeddings(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # 选择一个合适的嵌入模型
        input=texts
    )
    embeddings = [data['embedding'] for data in response['data']]
    return np.array(embeddings)

knowledge_embeddings = get_embeddings(knowledge_docs)

def retrieve_knowledge(user_input):
    # 生成用户输入的嵌入向量
    user_input_embedding = get_embeddings([user_input])[0]
    
    # 计算相似度并检索最相关的文档
    similarities = np.dot(knowledge_embeddings, user_input_embedding)
    most_similar_idx = np.argmax(similarities)
    
    return knowledge_docs[most_similar_idx]

def generate_response(user_id, conversation_history, user_input):
    # 更新对话历史
    conversation_history.append({"role": "user", "content": user_input})
    
    # 拼接对话历史用于上下文
    context = ' '.join([entry['content'] for entry in conversation_history])
    
    # 检索相关知识
    knowledge = retrieve_knowledge(user_input)
    
    # 创建 OpenAI ChatCompletion 请求
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
