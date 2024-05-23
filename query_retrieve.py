import uuid
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager, CallbackManagerForRetrieverRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

embeddings = OllamaEmbeddings(model="nomic-embed-text")
ollama_llm = 'qwen:7b'
model = Ollama(
    model=ollama_llm,
    temperature=0.1,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

run_manager = CallbackManagerForRetrieverRun(
    run_id=uuid.uuid4(),
    handlers=[StreamingStdOutCallbackHandler()],
    inheritable_handlers=[]
)

chromadb = Chroma(persist_directory="./chromadb/chroma_liver", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 6})


class Ranker(nn.Module):
    def __init__(self, bert_model):
        super(Ranker, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 使用[CLS]标记的表示
        score = self.fc(pooled_output)
        return score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
ranker = Ranker(bert_model).to(device)
ranker_model_path = "pairwise_ranker.pth"
ranker.load_state_dict(torch.load(ranker_model_path))
ranker.eval()


# 对重写问题利用ranker进行打分
def score_queries(queries, question_original, ranker, tokenizer):
    with torch.no_grad():
        scores = []
        for query in queries:
            if not query.strip():  # 去除空行的影响
                continue

            combined_text = f"{query}\n <------------------->\n {question_original}"
            combined_enc = tokenizer(combined_text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)

            # 提取 input_ids 和 attention_mask 并确保它们在正确的设备上
            input_ids = combined_enc['input_ids'].to(device)
            attention_mask = combined_enc['attention_mask'].to(device)

            # 计算重写问题score
            score = ranker(input_ids, attention_mask)
            scores.append(score)

    return scores


# 根据原始问题进行检索知识，问题重写之后利用ranker进行打分排序，然后返回top-k的重写问题进行向量库匹配
def retrieve_knowledge(question, retriever, ranker, tokenizer):
    multi_template = """我现在担任一名专注于医学领域的人工智能助手角色，特别是在处理肝病相关的咨询和信息提供方面。长篇的病例问题需要拆分成多个子问题
    来提高问题的清晰度和信息的可处理性。为了保证信息不丢失并提高问题的质量，请根据下述要求重写输入的问题：
    
    1. 根据给出问题的患者的入院情况，说明患者是什么原因住院，提取关键指标值和症状进行描述，并根据改写后的入院情况推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    2. 根据给出问题的患者的检查情况和检查结论，提取关键信息和指标值，重点关注检查结论，并且根据检查结论推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    3. 根据给出问题的患者的体格检查，不关注指标正常的信息，只提取异常的关键信息和指标值，并且根据改写后的体格检查信息推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    4. 根据给出问题的患者的查体信息，提取关键指标值和症状，尽可能保存原问题的表述，分析是否存在异常，并根据改写后的查体信息推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    5. 根据给出问题的患者的主诉信息，提取关键信息和各项指标值，并根据改写后的主诉信息推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    6. 根据给出问题的患者的现病史，提取患者的不适症状和指标值，说明患者已有的治疗方案和身体的变化，重点关注患者的现病史，并根据改写后的现病史信息推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    7. 根据给出问题的患者的既往史，提取患者曾经得过的疾病、不适症状和指标，并说明针对的治疗方案，包括服用的药物和做过手术，给出1~2种病的名称；然后给出对应的治理方案。
    8. 根据给出问题的患者个人史和家族史，提取患者以及家人可能患的疾病和关键信息指标，给出推测说明患者可能患有的疾病，给出1~2种病的名称；然后给出对应的治理方案。
    9. 根据给出问题的信息，如果还有没有分析到的项，请提取关键信息和指标值，重点关注异常情况、已说明的不适症状和对应的治疗方案，根据这些信息推测说明患者可能患有的疾病，给出21~2种病的名称；然后给出对应的治理方案。
    
    请按照要求进行重写问题，上述重写的问题使用准确的肝病医疗专业术语，确保问题重写符合医学专业知识和实践。每个重写问题都进行按序标号，并且把每个重写问题的所有信息放在一行，不能另起一行。
    生产10个及以上不同的、具有代替性的问题版本，每个问题都以不同的角度进行探讨。请使用中文输出，每个重写的问题之间用换行符分割，保证清晰易读。
    以下是待重写的示例问题，请按照上述指导原则进行修改和补充：{question}"""

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=multi_template,
    )

    retriever_llm = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=model,
        prompt=QUERY_PROMPT
    )

    rewritten_queries = retriever_llm.generate_queries(question, run_manager)  # 原始问题重写生成rewritten_queries
    rewritten_queries = [query for query in rewritten_queries if query.strip()]
    '''
    插入排序器 ,使用 Ranker 为重写的问题打分
    排序之后返回Top-k的重写问题保存在 reranked_queries中
    '''
    scores = score_queries(rewritten_queries, question, ranker, tokenizer)
    scored_queries = list(zip(scores, rewritten_queries))
    scored_queries.sort(reverse=True, key=lambda x: x[0])
    print(scored_queries)
    top_k_queries = [query for _, query in scored_queries[:3]]  # k是您想要的问题数量
    print(top_k_queries)
    retrieved_docs = retriever_llm.retrieve_documents(top_k_queries, run_manager)
    print(f"\nrecall num:" + str(len(retrieved_docs)))
    knowledge = ""
    for doc in retrieved_docs:
        knowledge += f"{doc.page_content}\n{doc.metadata}\n----------------------------------------\n"

    return knowledge


question = "请判断该病人是以下哪种疾病。病人检查如下：检查分类:MRI检查所见:肝脏外形光滑，各叶比例适中。肝实质反相位信号不均匀性减低，脂肪分数为21%。肝内外胆管无扩张，胆囊不大，底部壁厚，增强明显强化。      脾不大，实质信号未见异常。胰腺形态如常，实质信号未见异常，胰管无扩张。      双肾形态如常，双肾内可见类圆形T2WI高信号，增强无强化，较大者右肾长径约1.1cm。双肾盂及输尿管上段未见扩张，腔内未见异常信号。双侧肾上腺形态、信号未见异常。      腹腔及腹膜后间隙未见肿大淋巴结。检查结论:1、双肾小囊肿；       2、不均匀性中度脂肪肝；       3、胆囊底部壁增厚，腺肌症可能，请结合临床。入院情况：患者主因“发现脂肪肝5年余，发现右肝结节1年余”入院。查体：体温:36.5℃ 脉搏:69次/分 呼吸:18次/分 血压:129/75mmHg。腹软，呼吸运动正常，无脐疝、腹壁静脉曲张，无皮疹、色素沉着，未见胃肠型及蠕动波。腹壁柔软，右中腹压痛，无反跳痛、肌紧张。肝脏肋下2cm可触及，剑突下未触及，脾脏未触及。胆囊区无压痛，Murphy氏征阴性。振水音阴性。肝浊音界正常，肝区、肾区无叩击痛，移动性浊音阴性。肠鸣音正常，未闻及血管杂音。无静脉曲张，双下肢无水肿。主诉：发现脂肪肝5年余，发现右肝结节1年余现病史：患者5年余前体检发现脂肪肝，未规律复查，自诉肝功能正常。患者1年余前起进食油腻食物或辛辣刺激食物后出现右上腹及右中腹疼痛，为阵发性隐痛，可耐受，自诉右上腹可触及包块，表面凹凸不平。伴反酸、烧心、嗳气，伴活动后腹胀，无皮肤巩膜黄染、腹泻，无食欲不振、乏力，无恶心、呕吐，无呕血、黑便，无双下肢水肿等不适。患者自行服用蒲公英根泡水后诉反酸、烧心、嗳气症状较前好转，于当地医院完善腹部超声提示右肝结节，脂肪肝，后进一步于当地肿瘤医院完善检查提示右肝良性结节（未见报告），未予进一步诊治。患者近期仍有右上腹不适，餐后明显，为进一步诊治于我院门诊就诊，完善AFP 3.22ng/ml，PIVKA-II 16.29mAU/ml，ALT 19U/L，AST 16.8U/L，ALP 76U/L，GGT 17U/L，CHOL 6.13mmol/L，LDL-C 3.70mmol/L，门诊以右上腹不适原因待查收入我科。起病以来，患者精神、食欲、睡眠可，大便不成形，小便正常，体重无明显变化。既往史：否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。否认肝炎史、结核史、疟疾史。手术史：无，过敏史：无，输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。个人史：出生并久居于本地，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，否认饮酒史。否认去过疫区、否认接触过从疫区来的人。家族史：父母已逝，父亲因肝癌去世。兄弟姐妹3人，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。"
print(f"问题：{question}")

knowledge_recall = retrieve_knowledge(question, retriever, ranker, tokenizer)
print(f"找回知识：\n{knowledge_recall}")

'''


# before rag
print("Before RAG\n")
before_rag_template= "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model  | StrOutputParser()
print(before_rag_chain.invoke({"topic":query}))


#after rag
print("after rag\n")
after_rag_template = """
    You're helpful research assistant,who answers questions based upon chunks of text provided in context.
    If you don't know the answer,just say that you don't know,don't try to make up an answer.
    Please reply with just the detailed answer, and your sources.If you're unable to answer the question, don't list sources.

    Context: {context}
    ---
    Answer the question based on the above context: {question}
    """
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context":retriever,"question":RunnablePassthrough()}
    | after_rag_prompt
    | model
    | StrOutputParser()
)
print(after_rag_chain.invoke(query))


'''
