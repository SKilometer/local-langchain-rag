import json
import csv
import os
import uuid
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
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

run_manager = CallbackManagerForRetrieverRun(
    run_id=uuid.uuid4(),
    handlers=[StreamingStdOutCallbackHandler()],
    inheritable_handlers=[]
)

chromadb = Chroma(persist_directory="./chromadb/chroma_liver", embedding_function=embeddings)
retriever = chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 6})         # 每个问题取6个最相关的chunks


# evaluation 使用大模型进行打分输入（question, context）进行0-1打分
def evaluate_relevance(question, context):
    CATEGORICAL_TEMPLATE = """你现在是一名人工智能助手，你的任务是判断提供的上下文是否包含所给问题的相关信息，或者上下文是否可以用于回答问题。以下是数据:
    
        [Question]: {question}
        <--------------------------------------->
        [Context]: {context}
    
     如果上下文直接提供了问题的答案或包含关键信息和数据，可以明确或间接帮助回答问题，那么判断为“relevant”；
     如果上下文与问题无关，上下文不能够用来回答这个问题，或对回答问题没有帮助，或者信息不足以回答问题，那么判断为“irrelevant”。
     请仅使用单词“relevant”或“irrelevant”来表达你的判断，并且不应该包含除该单词之外的任何文本或字符，确保回答的简洁和准确。
     """
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=CATEGORICAL_TEMPLATE
    )

    llm_chain = prompt | model

    result = llm_chain.invoke({"question": question, "context": context}).strip().lower()
    if result == "relevant":
        return 1
    elif result == "irrelevant":
        return 0
    else:
        return 0  # 默认返回0以确保返回值为整数


query = "请判断该病人是以下哪种疾病。病人检查如下：入院情况：患者刘柃辉，男，50岁，主因“乏力纳差、眼黄尿黄4天”于2022-09-14入院。查体：体温:36.3℃ 脉搏:84次/分 呼吸:18次/分 血压:120/72mmHg。神清状可，皮肤轻度黄染，巩膜黄染，未见肝掌、蜘蛛痣。双肺呼吸音正常，未闻及干、湿性啰音及胸膜摩擦音。心律齐，无杂音。腹软，无压痛、反跳痛、肌紧张，未触及包块。肝脾肋下未触及。胆囊区无压痛，Murphy氏征阴性。移动性浊音阴性。肠鸣音正常，未闻及血管杂音。双下肢无水肿。主诉：乏力纳差、眼黄尿黄4天现病史：患者4天前无明显诱因出现眼黄、尿黄，伴皮肤黄染，伴乏力、纳差，无腹痛、腹泻，无恶心、呕吐，无双下肢水肿等不适。1天前就诊我科，查C11+C12:ALT)1058 U/L，AST)357.3 U/L，ALP)211 U/L，GGT)105 U/L，ALB)42.6 g/L，GLB)29.0 g/L，T-BIL)87.54 umol/L，D-BIL)47.16 umol/L，CHE)8.76 KU/L，Cr)104.1 umol/L；乙肝五项+丙肝+甲肝:未见异常；PT(A))92.00 %。腹部超声：肝内多发高回声结节，建议增强影像学检查，胆囊结石，胆囊壁多发胆固醇结晶。予患者口服保肝药物治疗，现患者为求进一步诊治收入我科。患者自发病以来，纳差，嗜睡，小便如上述，大便正常，近2周体重下降2kg。既往史：1月前体检发现肌酐升高，就诊我院予尿毒清口服半月（2022-8-15至2022-8-31）后出现腹部不适，自行停用，并自行服用枸橼酸莫沙必利片治疗10天（2022-9-1至2022-9-10）。糖尿病半年，空腹血糖最高19.6mmol/L，平素口服拜糖平1片 三餐中降糖治疗，自诉空腹血糖控制在5-6mmol/L。4年前发现肾结石，分别于4年前、2年前行碎石术。否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。否认肝炎史、结核史、疟疾史。手术史：无，过敏史：无，输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。个人史：出生并久居于本地，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，否认饮酒史。否认去过疫区、否认接触过从疫区来的人。家族史：父亲因心脏病过世，母亲健在，兄弟姐妹4人，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。"
context = """成人急性肝损伤诊疗急诊专家共识中国急救医学 2024年1月 第 44卷 第 1期 Chin J Crit Care Med Jan． 2024 ，Vol 44，No． 1中度和重度肝损伤 。黄疸持续时间 、TBiL水平和INＲ是ALI高风险的重要预测指标 。(证据等级B，推荐强度 A)3． 2． 2 ALI 的诊断分类评估推荐基于肝损伤生物化学异常模式的临床分型和Ｒ值进行诊断分类评估［41］。用ALT /ALP 来判断肝损伤类型 ，其总体符合率为 76%;而用来判断肝细胞损伤模式 ，其符合率则为 96%［42 － 43］。Ｒ =(ALT /ULN )/(ALP /ULN ){'page': 3, 'source': 'pdfs/成人急性肝损伤诊疗急诊专家共识.pdf'}"""
relevant_score = evaluate_relevance(query, context)
print("Relevant_score:", relevant_score)


# 将问题重写，得到重写问题进行评估打分，按照得分高低进行排序并且返回排好序的重写问题列表
def get_rank(question, retriever):
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
    # 问题重写
    rewritten_queries = retriever_llm.generate_queries(question, run_manager)
    print(f"\n{len(rewritten_queries)}")

    scores = []
    for rewritten_query in rewritten_queries:
        if not rewritten_query.strip():  # 去除空行的影响
            continue
        # 计算每个重写问题的得分
        total_score = 0
        unique_docs = retriever.invoke(rewritten_query)  # 每个重写问题找到top-k个chunks
        for doc in unique_docs:
            context = doc.page_content
            score = evaluate_relevance(question, context)
            if score is None:  # 处理score返回None的情况
                score = 0
            total_score += score
        print("total_socre:", total_score)
        scores.append((rewritten_query, total_score))  # [(q,s),(q,s),(q,s)...]

    scores.sort(key=lambda x: x[1], reverse=True)  # 返回重排的（q,s）

    return scores


# 读取带有原始问题的json文件，读取question项，进行打分排序，将得到的排序和原始问题一起返回到csv文件
def process_jsonl(jsonl_filename, csv_filename):
    directory = os.path.dirname(csv_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(jsonl_filename, 'r', encoding='utf-8') as jsonl_file, open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Original Question", "Sorted Rewritten Questions and Scores"])

        for line in jsonl_file:
            data = json.loads(line)
            question = data.get("question")

            if question:
                sorted_scores = get_rank(question, retriever)
                sorted_rewritten_queries_and_scores = [(q, s) for q, s in sorted_scores]
                writer.writerow([question, sorted_rewritten_queries_and_scores])


jsonl_filename = 'data_clean/liver_question_answer.jsonl'
csv_filename = 'sorted/sorted_liver_questions.csv'

process_jsonl(jsonl_filename, csv_filename)

'''

query = "请判断该病人是以下哪种疾病。病人检查如下：入院情况：患者刘柃辉，男，50岁，主因“乏力纳差、眼黄尿黄4天”于2022-09-14入院。查体：体温:36.3℃ 脉搏:84次/分 呼吸:18次/分 血压:120/72mmHg。神清状可，皮肤轻度黄染，巩膜黄染，未见肝掌、蜘蛛痣。双肺呼吸音正常，未闻及干、湿性啰音及胸膜摩擦音。心律齐，无杂音。腹软，无压痛、反跳痛、肌紧张，未触及包块。肝脾肋下未触及。胆囊区无压痛，Murphy氏征阴性。移动性浊音阴性。肠鸣音正常，未闻及血管杂音。双下肢无水肿。主诉：乏力纳差、眼黄尿黄4天现病史：患者4天前无明显诱因出现眼黄、尿黄，伴皮肤黄染，伴乏力、纳差，无腹痛、腹泻，无恶心、呕吐，无双下肢水肿等不适。1天前就诊我科，查C11+C12:ALT)1058 U/L，AST)357.3 U/L，ALP)211 U/L，GGT)105 U/L，ALB)42.6 g/L，GLB)29.0 g/L，T-BIL)87.54 umol/L，D-BIL)47.16 umol/L，CHE)8.76 KU/L，Cr)104.1 umol/L；乙肝五项+丙肝+甲肝:未见异常；PT(A))92.00 %。腹部超声：肝内多发高回声结节，建议增强影像学检查，胆囊结石，胆囊壁多发胆固醇结晶。予患者口服保肝药物治疗，现患者为求进一步诊治收入我科。患者自发病以来，纳差，嗜睡，小便如上述，大便正常，近2周体重下降2kg。既往史：1月前体检发现肌酐升高，就诊我院予尿毒清口服半月（2022-8-15至2022-8-31）后出现腹部不适，自行停用，并自行服用枸橼酸莫沙必利片治疗10天（2022-9-1至2022-9-10）。糖尿病半年，空腹血糖最高19.6mmol/L，平素口服拜糖平1片 三餐中降糖治疗，自诉空腹血糖控制在5-6mmol/L。4年前发现肾结石，分别于4年前、2年前行碎石术。否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。否认肝炎史、结核史、疟疾史。手术史：无，过敏史：无，输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。个人史：出生并久居于本地，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，否认饮酒史。否认去过疫区、否认接触过从疫区来的人。家族史：父亲因心脏病过世，母亲健在，兄弟姐妹4人，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。"
context = """成人急性肝损伤诊疗急诊专家共识中国急救医学 2024年1月 第 44卷 第 1期 Chin J Crit Care Med Jan． 2024 ，Vol 44，No． 1中度和重度肝损伤 。黄疸持续时间 、TBiL水平和INＲ是ALI高风险的重要预测指标 。(证据等级B，推荐强度 A)3． 2． 2 ALI 的诊断分类评估推荐基于肝损伤生物化学异常模式的临床分型和Ｒ值进行诊断分类评估［41］。用ALT /ALP 来判断肝损伤类型 ，其总体符合率为 76%;而用来判断肝细胞损伤模式 ，其符合率则为 96%［42 － 43］。Ｒ =(ALT /ULN )/(ALP /ULN ){'page': 3, 'source': 'pdfs/成人急性肝损伤诊疗急诊专家共识.pdf'}"""

relevant_score = evaluate_relevance(query, context)
print("\Relevant_score:", relevant_score)
sorted = get_rank(query, retriever)
print(f"\n{sorted}")


sorted = get_rank(query, retriever)
sorted_rewritten_queries = [q for q, s in sorted]

csv_filename = 'sorted_questions.csv'
save_to_csv(csv_filename, query, sorted_rewritten_queries)

def read_csv(csv_filename):
    with open(csv_filename, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Skip the header row
        data = []

        for row in reader:
            original_question = row[0]
            rewritten_queries_and_scores = eval(row[1])  # Use eval to parse the list of tuples

            for q, s in rewritten_queries_and_scores:
                data.append({
                    "original_question": original_question,
                    "q": q,
                    "s": s
                })
                    
        return data

original_question, second_ranked_question = read_from_csv(csv_filename)
print(f"Original Question: {original_question}")
print(f"Second Ranked Question: {second_ranked_question}")
'''
