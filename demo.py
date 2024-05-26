import gradio as gr
from load_vistral import VistralChat

# Initialize the chatbot
vistral_chat = VistralChat()

def chat_with_bot(user_input):
    response = vistral_chat.conversation(user_input)
    return response

iface = gr.Interface(
    fn=chat_with_bot,
    inputs= gr.Textbox(lines=5, label="Câu hỏi"),
    outputs=gr.Textbox(lines=5, label="Câu trả lời"),
    title="MISA Chatbot",
    description="Hỏi bất kỳ câu hỏi nào và nhận câu trả lời từ chatbot MISA.",
)

iface.launch(share=True)
