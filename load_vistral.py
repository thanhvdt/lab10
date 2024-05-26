import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from load_link import get_google_search_results

token = "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"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

SYSTEM_PROMPT_0 = """
Bạn là một trợ lý ẩm thực người Việt. Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch. Với mỗi câu hỏi, hệ thống cần cung cấp: công thức, nguyên liệu, và các bước thực hiện chi tiết.
"""

SYSTEM_PROMPT_1 = """
Bạn là một trợ lý ẩm thực người Việt. Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch. Vui lòng cung cấp một công thức tổng quát và liệt kê đầy đủ các nguyên liệu quan trọng, ví dụ: nguyên liệu là gì, cách chế biến như thế nào.
"""

SYSTEM_PROMPT_2 = """
Bạn là một trợ lý ẩm thực người Việt. Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch. Vui lòng trả ra nội dung gồm các bước chế biến, thời gian nấu, và các lưu ý quan trọng để món ăn thành công.
"""

SYSTEM_PROMPT = """
Bạn là một trợ lý ẩm thực người Việt. Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
"""

def check_content(text):
    class1 = ["Tóm tắt", "tóm tắt"] 
    class2 = ["so sánh", "So sánh", "Điểm mới", "điểm mới"]
    for phrase in class1:
        if phrase in text: 
            return 1
    for phrase in class2:
        if phrase in text: 
            return 2
    return 0

class VistralChat:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Viet-Mistral/Vistral-7B-Chat', use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            'Viet-Mistral/Vistral-7B-Chat',
            token=token,
            torch_dtype=torch.bfloat16,  # change to torch.float16 if you're using V100
            device_map="auto",
            use_cache=True,
            quantization_config=bnb_config
        )
    def conversation(self, text):
        prompt_class = check_content(text)
        if prompt_class == 0:
            conversation = [{"role": "system", "content": SYSTEM_PROMPT }]
        elif prompt_class == 1:
            conversation = [{"role": "system", "content": SYSTEM_PROMPT_1 }]
        elif prompt_class == 2:
            conversation = [{"role": "system", "content": SYSTEM_PROMPT_2 }]
        else:
            conversation = [{"role": "system", "content": SYSTEM_PROMPT }]

        conversation.append({"role": "user", "content": text })
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        self.model.eval()
        out_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=4096,
            do_sample=False,
            top_p=1,
            temperature=0,
            repetition_penalty=1.05,
        )
        assistant = self.tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()
        return assistant + "\n\n" + get_google_search_results(text)

if __name__ == "__main__":
    vistral_chat = VistralChat()
    user_input = input("Bạn muốn hỏi gì? ")
    response = vistral_chat.conversation(user_input)
    print("Phản hồi:", response)
