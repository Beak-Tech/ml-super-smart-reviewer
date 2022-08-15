from transformers import AutoTokenizer, pipeline, AutoModelForQuestionAnswering

qa_path = "deepset/roberta-base-squad2"
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_path)
qa_pip=pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer, top_k=3)

def prediction(question, context):
    qa_input = {
        'question' :  question,
        'context' : context 
    }
    result = qa_pip(qa_input)
    return result
