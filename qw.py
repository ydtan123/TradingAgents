#!/usr/bin/env python3
import os
import argparse
from prompts import STOCK_LIST
from pathlib import Path

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run financial analysis with OpenAI.")
    parser.add_argument("--prompt", "-p", type=str, default="", help="The prompt to use for the analysis.")
    parser.add_argument("--file", "-f", type=str, default="", help="The file to use for the analysis.")
    args = parser.parse_args()

    try:
        client = OpenAI(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
            # 新加坡和北京地域的API Key不同。获取API Key: https://help.aliyun.com/zh/model-studio/get-api-key
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        print(args.file)
        if args.prompt != "":
            full_prompt = args.prompt
        elif args.file != "":
            print("Uploading file for analysis...")
            file_object = client.files.create(file=Path("sample.pdf"), purpose="batch")
            print(file_object.model_dump_json())
            assistant = client.beta.assistants.create(
                name="PDF Analyzer",
                instructions="你是一个文档分析助手，请根据用户提供的 PDF 内容回答问题。",
                model="gpt-4-turbo",  # 或 gpt-4o
                tools=[{"type": "retrieval"}],  # 启用文件检索能力
                file_ids=[file_object.id])  # 关联上传的文件
            thread = client.beta.threads.create()

            # 添加用户消息（可以不包含文件内容，因为 assistant 已绑定文件）
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="请总结这份 PDF 的主要内容。"
            )
            print(message.model_dump_json())
        else:
            full_prompt = STOCK_LIST
        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a financial analyst.'},
                {'role': 'user', 'content': full_prompt}
                ]
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")