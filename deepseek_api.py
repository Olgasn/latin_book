import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langgraph.graph import StateGraph, END


# Установка ключа DEEPSEEK API . Рекомендуется использовать переменные окружения.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:    
    raise ValueError("Необходимо установить переменную окружения DEEPSEEK_API_KEY")

# Подключение к удалённому серверу OpenAI
model_name="deepseek-chat"
llama = ChatOpenAI(
    model=model_name,
    temperature=0,
    max_tokens=6000,
    top_p=0.7,
    stream_usage=True,
    timeout=10000,
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com",
    max_retries=10,
    
    organization="University GSTU",
    # other params...
)

# Функция для загрузки промтов из файлов
def load_prompt_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {filename} не найден.")

# Фабрика для создания цепочек
def create_chain(prompt_template, llm, output_parser):
    return prompt_template | llm | output_parser



# Память для записи взаимодействий
interaction_memory = []


# Агент: Менеджер
manager_prompt = PromptTemplate(
    template="""

    Вы специалист, создающий учебное пособие для студентов университета для изучения дисциплины "Латинский язык". 
    Вы исключительно хорошо владеете русским, английским и латинским языками.
    Настрой среду для дальнейшего взаимодействия между тремя экспертами для ответа на каждый вопрос из списка.
    Первый эксперт (expert) получает на вход один их вопросов из списка и отвечает на него со всей полнотой, правильностью и ясностью.
    Второй эксперт (reviewer) анализирует ответ первого эксперта на этот вопрос и дает рекомендации по улучшению качества и полноты ответа.
    Третий эксперт (finalizer) на основании вопроса, ответа первого эксперта и рекомендаций второго эксперта, формирует итоговый ответ и записывает его в файл.
    После этого идет переход к новому вопросу из списка и все повторяется до тех пор, пока не будут получены полные правильные и исчерпывающие ответы на все вопросы.

    Ответ:


    """,
    #input_variables=["question"],
)
manager_chain = manager_prompt | llama | StrOutputParser()

# Агент: Преподаватель
expert_prompt = PromptTemplate(
    template="""


    Вы эксперт по латинскому языку. 
    Дайте ответ на поставленный вопрос, основываясь на информации из признанных компетентных источников, учебников и книг.
    Вопрос: {question}
    Ответ на вопрос должен быть подробным, точным и полезным, обязательно содержать примеры, позволяющие глубже студенту понять изложенное.

    Ответ:
    """,
    input_variables= ["question"],
)
expert_chain =  expert_prompt | llama | StrOutputParser()


# Агент: Оценщик
reviewer_prompt = PromptTemplate(
    template="""

    Вы опытный эксперт по латинскому языку и другим языкам.
    Оцените предоставленный ответ на вопрос на: правильность утверждений; полноту изложения вопроса; наличие примеров. Выставьте оценку ответу (assessment) от 1 до 10 (Оценка: ...).
    Напишите в ответе (answer) рекомендации, содержащие предложения по уточнениям, правкам или дополнениям, если это требуется.

    Вопрос: {question}
    Ответ: {answer}
    Assessment:
 

    """,
    input_variables=["question", "answer"],
)
reviewer_chain = reviewer_prompt | llama | StrOutputParser()


# Агент: Супер эксперт
finalizer_prompt = PromptTemplate(
    template="""


    Вы исключительно опытный эксперт по латинскому языку и другим языкам. Вы обладаете большими познаниями в смежных областях. 
    Ваша задача - на основе вопроса, ответа на него и отзыва оценщика сформировать наиболее полный и совершенный ответ с объемом не меньше первоначального объема.
    Вы исключительно опытный эксперт по базам данных, системам управления базами данных и автоматизированным информационным системам, а также по разработке программных приложений. Вы обладаете большими познаниями в смежных областях. 
    Ваша задача - на основе вопроса, ответа на него и отзыва оценщика сформировать наиболее полный и совершенный ответ с объемом не меньше первоначального объема
    Вопрос: {question}
    Ответ: {answer}
    Замечания: {feedback}
    Answer:

    """,
    input_variables= ["question", "answer", "feedback"],
)
finalizer_chain =  finalizer_prompt | llama | StrOutputParser()


# Процесс работы
class GraphState(dict):
    """
    Represents the state of the process.

    Attributes:
        question: Current question being processed.
        expert_answer: Answer provided by the teacher.
        assessment: Score from the reviewer
        reviewer_feedback: Feedback from the reviewer.
        final_answer: Finalized answer after feedback.
    """
    question: str
    expert_answer: str
    assessment: str
    reviewer_feedback: str
    final_answer: str

# Узлы графа

def manager_node(state):
    question = state["question"]
    # возможна дополнительная обработка вопросов, например, перевод на другой язык
    answer=manager_chain.invoke({"question": question})
    interaction_memory.append({"role": "manager", "answer": answer})
    return {"question": question, "manager_answer": answer}

def expert_node(state):
    question = state["question"]
    answer = expert_chain.invoke({"question": question})
    interaction_memory.append({"role": "expert", "question": question, "answer": answer})
    return {"question": question,"expert_answer": answer}

def reviewer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    feedback = reviewer_chain.invoke({"question": question, "answer": answer})
    assessment=feedback.split(":")[1].split("\n")[0]
    if  assessment.strip().isdigit():
        assessment  =  int(assessment.strip())
    else:
        assessment = int(assessment.split("/")[0].replace("*", "").strip())
    interaction_memory.append({"role": "reviewer", "question": question, "assessment":assessment, "feedback": feedback})
    return {"question": question, "expert_answer": answer, "assessment": assessment, "reviewer_feedback": feedback}


def finalizer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    feedback = state["reviewer_feedback"]
    assessment=state["assessment"]
    if int(assessment)>9:
        final_answer = answer
    else:
        final_answer = finalizer_chain.invoke({"question": question, "answer": answer, "feedback":feedback})
    interaction_memory.append({"role": "finalizer",  "answer": answer, "assessment":assessment, "final_answer":  final_answer})
    return {"question": question, "expert_answer": answer, "assessment": assessment, "reviewer_feedback": feedback, "final_answer": final_answer}

def finalize_answer(state):
    question = state["question"]
    assessment = state["assessment"]
    feedback = state["reviewer_feedback"]
    expert_answer = state["expert_answer"]
    # возможна дополнительная обработка ответов, например, перевод на другой язык
    final_answer = state["final_answer"]
    return {"question": question, "expert_answer": expert_answer, "assessment": assessment, "reviewer_feedback": feedback, "final_answer": final_answer}

# Построение графа
workflow = StateGraph(GraphState)
workflow.add_node("manager", manager_node)
workflow.add_node("expert", expert_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("finalizer", finalizer_node)
workflow.add_node("finish", finalize_answer)
workflow.set_entry_point("manager")

workflow.add_edge("manager", "expert")
workflow.add_edge("expert", "reviewer")
workflow.add_edge("reviewer", "finalizer")
workflow.add_edge("finalizer", "finish")
workflow.add_edge("finish", END)

# Компиляция графа
agent_workflow = workflow.compile()

def run_agent(question):
    state = {"question": question}
    output = agent_workflow.invoke(state)
    with open(model_name+"_tutorial_output.txt", "a", encoding="utf-8") as f:
        f.write(f"Вопрос: {state['question']}\nОтвет: {output['final_answer']}\n\n")
    return output

def load_questions_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            questions = [line.strip() for line in file.readlines()]
        return questions
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return []

def save_to_json(interaction_memory, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(interaction_memory, file, ensure_ascii=False, indent=4)

# Тестовые вопросы
questions = load_questions_from_file('questions.txt')

for q in questions:
    print(run_agent(q))

save_to_json(interaction_memory, model_name+'_log.json')




