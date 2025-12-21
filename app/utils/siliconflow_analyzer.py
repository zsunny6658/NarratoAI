"""
SiliconFlow视觉分析器
使用OpenAI兼容接口调用SiliconFlow的DeepSeek OCR模型
@author: Sunny
"""

import json
from typing import List, Union, Dict
import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import asyncio
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential
from openai import OpenAI
import PIL.Image
import base64
import io
import re


class SiliconFlowAnalyzer:
    """SiliconFlow视觉分析器类"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", api_key: str = None, base_url: str = None):
        """
        初始化SiliconFlow视觉分析器
        
        Args:
            model_name: 模型名称，默认使用 deepseek-ai/DeepSeek-OCR
            api_key: SiliconFlow API密钥
            base_url: API基础URL，如果为None则使用默认值
        """
        if not api_key:
            raise ValueError("必须提供API密钥")

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.siliconflow.cn/v1"

        # 配置API客户端
        self._configure_client()

    def _configure_client(self):
        """
        配置API客户端
        使用OpenAI SDK来调用 SiliconFlow API
        """
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"配置SiliconFlow API，端点: {self.base_url}, 模型: {self.model_name}")
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {str(e)}")
            raise


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _generate_content_with_retry(self, prompt, batch):
        """
        使用重试机制调用SiliconFlow API
        
        Args:
            prompt: 提示词
            batch: 图片对象
            
        Returns:
            API响应结果
        """
        try:
            return await self._generate_with_siliconflow_api(prompt, batch)
        except Exception as e:
            logger.warning(f"SiliconFlow API请求异常: {str(e)}")
            raise


    async def _generate_with_siliconflow_api(self, prompt, batch):
        """
        使用SiliconFlow API生成内容
        
        Args:
            prompt: 提示词
            batch: 图片对象
            
        Returns:
            API响应结果
        """
        # 将图片转换为base64
        # 将PIL图片转换为字节流
        """使用OpenAI兼容接口生成内容"""
        # 将PIL图片转换为base64编码
        image_contents = []
        for img in batch:
            # 将PIL图片转换为字节流
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=85)
            img_bytes = img_buffer.getvalue()
            
            # 转换为base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # 构建OpenAI格式的消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_contents
                ]
            }
        ]

        logger.info(f"发送请求到 SiliconFlow API: {self.base_url}")
        logger.info(f"模型: {self.model_name}")
        logger.info(f"提示词: {prompt}")
        logger.info(f"图片数量: {len(batch)}")
        
        # 调用OpenAI兼容接口
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=4000,
            temperature=1.0
        )

        logger.info(f"收到响应: {response.choices[0].message.content}")
        
        return response.choices[0].message.content

    async def analyze_images(self,
                           images: List[Union[str, Path, PIL.Image.Image]],
                           prompt: str,
                           batch_size: int = 10) -> List[str]:
        """
        分析图片并返回结果
        
        Args:
            images: 图片路径列表或PIL图片对象列表
            prompt: 分析提示词
            batch_size: 批处理大小
            
        Returns:
            分析结果列表，每个元素为包含结果或错误信息的字典
        """
        logger.info(f"开始使用 SiliconFlow ({self.model_name}) 分析 {len(images)} 张图片")

        # 加载图片
        loaded_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                try:
                    pil_img = PIL.Image.open(img)
                    # 调整图片大小以优化性能
                    if pil_img.size[0] > 1024 or pil_img.size[1] > 1024:
                        pil_img.thumbnail((1024, 1024), PIL.Image.Resampling.LANCZOS)
                    loaded_images.append(pil_img)
                except Exception as e:
                    logger.error(f"加载图片失败 {img}: {str(e)}")
                    continue
            elif isinstance(img, PIL.Image.Image):
                loaded_images.append(img)
            else:
                logger.warning(f"不支持的图片类型: {type(img)}")
                continue

        if not loaded_images:
            raise ValueError("没有有效的图片可以分析")
        
        # TODO 按照入参，批处理来改造
        results = []
        total_images = len(images)
        
        # 批次大小默认为1个来处理
        for i in range(0, total_images, batch_size):
            batch = loaded_images[i:i + batch_size]
            try:
                response = await self._generate_content_with_retry(prompt, batch)
                result_dict = {
                                'batch_index': i // batch_size,
                                'images_processed': len(batch),
                                'response': response,
                                'model_used': self.model_name
                            }
                logger.info(f"图片处理结果: {result_dict}")
                results.append(result_dict)
                # 添加延迟以避免API限流
                if i + batch_size < len(loaded_images):
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"图片 {batch[0].filename} 处理失败: {str(e)}")
        
        logger.info(f"完成图片分析，共处理 {len(results)} 张图片")
        return results
