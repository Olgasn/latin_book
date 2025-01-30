import json
import os
import fitz 
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


from langchain_google_genai import ChatGoogleGenerativeAI
# Установка ключа API Gemini. Рекомендуется использовать переменные окружения.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Необходимо установить переменную окружения GOOGLE_API_KEY")

gemini_model_name="gemini-2.0-flash-exp"
# Инициализация модели Gemini. Укажите нужную модель.
gemini = ChatGoogleGenerativeAI(
    model=gemini_model_name,
    temperature=0,
    max_tokens=1000000,        
    max_output_tokens=59000, # Используйте max_output_tokens для Gemini
)



# Установка ключа DEEPSEEK API . Рекомендуется использовать переменные окружения.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:    
    raise ValueError("Необходимо установить переменную окружения DEEPSEEK_API_KEY")


model_name="deepseek-chat"
llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    max_tokens=65500,
    top_p=0.7,
    stream_usage=True,
    timeout=10000,
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com",
    max_retries=10,
    organization="University GSTU",
    # other params...
)


# Память для записи взаимодействий с LLM
interaction_memory = []

# Шаблоны промптов

start_prompt = PromptTemplate(
    template="""
    Вы исключительно опытный эксперт по латинскому языку и другим языкам. Вы обладаете большими познаниями в смежных областях.
    Переформулируй и укрупни прилагаемые вопросы по учебной дисциплине "Латинский язык", так чтобы осталось 14 емких и не пересекающихся между собой вопросов, отражающих все содержание учебной дисциплины.
    Сделай выборку ключевой информации из учебных пособий для ответов на укрупненные вопросы, информация не должна дублироваться. 
    Выборка должна быть достаточно подробной и содержательной.
    Вопросы по учебной дисциплине "Латинский язык": {questions}
    Информация из учебных пособий: {context}

    Сформированный вами ответ будет учитываться экспертами при формировании ответов на каждый из вопросов.

    Ответ:
    """,
    input_variables=["questions", "context"],

)


manager_prompt = PromptTemplate(
    template="""{history}

    Вы специалист, создающий учебное пособие для студентов университета для изучения дисциплины "Латинский язык". 
    Вы исключительно хорошо владеете русским, английским и латинским языками.
    Настрой среду для дальнейшего взаимодействия между тремя экспертами для ответа на каждый вопрос из списка.
    Первый эксперт (expert) получает на вход один их вопросов из списка и отвечает на него со всей полнотой, правильностью и ясностью.
    Второй эксперт (reviewer) анализирует ответ первого эксперта на этот вопрос и дает рекомендации по улучшению качества и полноты ответа.
    Третий эксперт (finalizer) на основании вопроса, ответа первого эксперта и рекомендаций второго эксперта, формирует итоговый ответ и записывает его в файл.
    После этого идет переход к новому вопросу из списка и все повторяется до тех пор, пока не будут получены полные правильные и исчерпывающие ответы на все вопросы.
 

    Ответ:
    
    """,
    input_variables=["history"],

)

expert_prompt = PromptTemplate(
    template="""

    Вы эксперт по латинскому языку. 
    Дайте ответ на поставленный вопрос, основываясь на информации из признанных компетентных источников, учебников и книг.
    Вопрос: {question}
    Дополнительная информация для ответа, полученная из обзора рекомендуемых источников: {history}
    
    Ответ на вопрос должен быть подробным, точным и полезным, содержать примеры, позволяющие глубже студенту понять изложенное.
    
    Ответ:
    """,
    input_variables=["question", "history"],
)

reviewer_prompt = PromptTemplate(
    template="""
    Вы опытный эксперт по латинскому языку и другим языкам.
    Оцените предоставленный ответ на вопрос на: правильность утверждений; полноту изложения вопроса; наличие примеров. 
    Выставьте оценку ответу (assessment) от 1 до 10 (Оценка: ...).
    Напишите в ответе (answer) рекомендации, содержащие предложения по уточнениям, правкам или дополнениям, если это требуется.

    Вопрос: {question}
    Ответ: {answer}
    Дополнительная информация для ответа, полученная из обзора рекомендуемых источников: {history}
    
    Assessment:
    """,
    input_variables=["question", "answer", "history"],
)

finalizer_prompt = PromptTemplate(
    template="""
    Вы исключительно опытный эксперт по латинскому языку и другим языкам. Вы обладаете большими познаниями в смежных областях. 
    Ваша задача - на основе вопроса, ответа на него и отзыва оценщика сформировать наиболее полный и совершенный ответ с объемом не меньше первоначального объема.
    
    Вопрос: {question}
    Ответ: {answer}
    Замечания: {feedback}

    Ответ:
    """,
    input_variables=["question", "answer", "feedback"],
)


# Фабрика для создания цепочек
def create_chain(prompt_template, llm, output_parser, input_data):
    chain = prompt_template | llm | output_parser
    output = chain.invoke(input_data)
    return output

# Создание цепочек с памятью
start_chain = start_prompt | gemini | StrOutputParser() # Начальная цепочка
manager_chain = manager_prompt | llm | StrOutputParser()
expert_chain = expert_prompt| llm | StrOutputParser()
reviewer_chain = reviewer_prompt| llm | StrOutputParser()
finalizer_chain = finalizer_prompt| llm | StrOutputParser()



# Процесс работы
class GraphState(dict):
    """
    Represents the state of the process.

    Attributes:
        history:
        question: Current question being processed.
        expert_answer: Answer provided by the teacher.
        assessment: Score from the reviewer
        reviewer_feedback: Feedback from the reviewer.
        final_answer: Finalized answer after feedback.
    """
    history: str=""
    question: str
    expert_answer: str
    assessment: str
    reviewer_feedback: str
    final_answer: str



# Узлы графа

def manager_node(state):
    question = state["question"]
    history= state["history"]
    # возможна дополнительная обработка вопросов, например, перевод на другой язык
    answer=manager_chain.invoke({"question": question, "history": history})
    interaction_memory.append({"role": "manager", "answer": answer})
    return {"question": question, "manager_answer": answer, "history": history}

def expert_node(state):
    question = state["question"]
    history= state["history"]
    answer = expert_chain.invoke({"question": question, "history":history})
    interaction_memory.append({"role": "expert", "question": question, "answer": answer, "history": history})
    return {"question": question,"expert_answer": answer}

def reviewer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    history= state["history"]
    feedback = reviewer_chain.invoke({"question": question, "answer": answer, "history": history})
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
    state = {"question": question,"history":history}
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

def save_to_text_file(context, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:  # Открываем файл для записи
            file.write(context)  # Записываем текст в файл
    except Exception as e:
        print(f"Произошла ошибка при записи в файл {filename}: {e}")


# Извлекает текст из PDF-файла
def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF-файла."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except fitz.FileDataError:
        print(f"Ошибка: Не удалось открыть файл {pdf_path}. Возможно, файл поврежден или не существует.")
        return None  # Важно возвращать None при ошибке
    except Exception as e:
        print(f"Произошла ошибка при обработке PDF: {e}")
        return None


# Извлечение текста из всех PDF-файлов в папке
pdf_texts = []
pdf_folder = Path("./books")  # Укажите путь к папке с PDF-файлами
if not pdf_folder.exists():
    print(f"Папка {pdf_folder} не найдена. Создайте папку 'pdfs' и поместите туда PDF файлы")
    exit()

for pdf_file in pdf_folder.glob("*.pdf"):  # Ищем все файлы с расширением .pdf
    pdf_text = extract_text_from_pdf(pdf_file)
    if pdf_text:
        pdf_texts.append(pdf_text)
    else:
        print(f"Файл {pdf_file.name} будет пропущен")
    
if not pdf_texts: #проверка, что текст из pdf был загружен
    print("Не удалось загрузить текст ни из одного PDF файла. Работа будет продолжена без контекста")
    context = ""
else:
    # Объединение текста из PDF в общий контекст
    context = "\n".join(pdf_texts)

save_to_text_file(context, "book_context.txt")



# Тестовые вопросы
questions = load_questions_from_file('questions.txt')


# Начало процесса ответов
history=start_chain.invoke({"questions":questions,"context":context})
save_to_text_file(history, "context_from_llm.txt")
interaction_memory.append({"role": "start", "answer": history})

for q in questions:
    print(run_agent(q))

save_to_json(interaction_memory, model_name+'_log.json')