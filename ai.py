import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 加载 .env 文件
load_dotenv()


class SmartTranslator:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件或环境变量设置。")

        self.model = ChatDeepSeek(
            model="deepseek-v4-flash",
            api_key=api_key,
            temperature=0.3,
        )

    def translate(self, text: str, target_lang: str = "中文", style: str = "正式") -> str:
        """翻译文本"""
        system_prompt = f"""你是一个专业的翻译助手。
任务:
1. 自动检测输入文本的语言
2. 翻译成{target_lang}
3. 使用{style}风格
4. 如果有专业术语，在翻译后用括号标注原文
输出格式:
【原语言】: XXX
【翻译】: XXX
【术语解释】: (如果有)
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ]
        response = self.model.invoke(messages)
        return response.content


def main():
    translator = SmartTranslator()

    print("\n进入交互模式（输入 'quit' 退出）\n")

    while True:
        text = input("请输入要翻译的文本: ").strip()

        # 退出条件
        if text.lower() in ("quit", "exit", "q"):
            print("已退出交互模式。")
            break

        # 空输入跳过
        if not text:
            print("输入为空，请重新输入。\n")
            continue

        # 目标语言，默认中文
        target = input("目标语言（默认中文）: ").strip() or "中文"

        # 翻译风格，默认正式
        style = input("翻译风格（正式/口语/文学，默认正式）: ").strip() or "正式"

        print("\n翻译中...\n")

        try:
            result = translator.translate(text, target, style)
            print(result)
        except Exception as e:
            print(f"翻译出错: {e}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()