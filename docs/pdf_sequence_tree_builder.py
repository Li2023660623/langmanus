#!/usr/bin/env python3
"""
PDF序号识别和树形结构构建工具
按照原有顺序和结构输出树形的序号树
"""

import re
import json
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SequenceNode:
    """序号节点"""
    text: str
    page: int
    rect: List[float]
    sequence_type: str
    value: str
    level: int
    children: List['SequenceNode']

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "text": self.text,
            "page": self.page,
            "rect": self.rect,
            "sequence_type": self.sequence_type,
            "value": self.value,
            "level": self.level,
            "children": [child.to_dict() for child in self.children]
        }


class PDFSequenceTreeBuilder:
    """PDF序号树形结构构建器"""

    def __init__(self):
        """初始化"""
        # 序号正则模式，按优先级排序
        self.patterns = [
            # 中文数字序号
            (r'^\s*[’‘“”\"\']*([一二三四五六七八九十]+)\s*、\s*', 'chinese_dot'),  # 匹配：一、二、三、等中文数字后跟顿号
            (r'^\s*[’‘“”\"\']*([一二三四五六七八九十]+)\s*[.]\s*', 'chinese_period'),  # 匹配：一.二.三.等中文数字后跟点号
            (r'^\s*[’‘“”\"\']*（([一二三四五六七八九十]+)）\s*', 'chinese_paren'),  # 匹配：（一）（二）（三）等中文数字带中文括号
            (r'^\s*[’‘“”\"\']*\(([一二三四五六七八九十]+)\)\s*', 'chinese_bracket'),  # 匹配：(一)(二)(三)等中文数字带英文括号

            # 阿拉伯数字序号
            (r'^\s*[’‘“”\"\']*(\d+)\s*[、]\s*', 'arabic_dot'),  # 匹配：1、2、3、等数字后跟顿号
            (r'^\s*[’‘“”\"\']*(\d+)\s*[.]\s*(?!\d)', 'arabic_period'),  # 匹配：1、2、3、等数字后跟点号后面不是数字的
            (r'^\s*[’‘“”\"\']*(\d+)\s*）\s*', 'arabic_right_paren'),  # 匹配：1）2）3）等数字后跟右括号
            (r'^\s*[’‘“”\"\']*（(\d+)）', 'arabic_paren'),  # 匹配：（1）（2）（3）等数字带中文括号
            (r'^\s*[’‘“”\"\']*\((\d+)\)', 'arabic_bracket'),  # 匹配：(1)(2)(3)等数字带英文括号
            # (r'^\s*[\[【](\d+)[\]】]\s*', 'arabic_square'),  # 匹配：[1][2]或【1】【2】等数字带方括号

            # 多级数字序号
            # (r'^\s*(\d+\.\d+)\s*[、]\s*', 'multi_level_2'),  # 匹配：1.1、1.2、等二级数字序号
            # (r'^\s*(\d+\.\d+\.\d+)\s*[、]\s*', 'multi_level_3'),  # 匹配：1.1.1、1.1.2、等三级数字序号
            # (r'^\s*(\d+\.\d+\.\d+\.\d+)\s*[、]\s*', 'multi_level_4'),  # 匹配：1.1.1.1、等四级数字序号

            # 字母序号
            (r'^\s*([A-Z])\s*[、]\s*', 'letter_upper_dot'),  # 匹配：A、B、C、等大写字母后跟顿号
            (r'^\s*([a-z])\s*[、]\s*', 'letter_lower_dot'),  # 匹配：a、b、c、等小写字母后跟顿号
            (r'^\s*（([A-Z])）\s*', 'letter_upper_paren'),  # 匹配：（A）（B）等大写字母带中文括号
            (r'^\s*（([a-z])）\s*', 'letter_lower_paren'),  # 匹配：（a）（b）等小写字母带中文括号
            (r'^\s*\(([A-Z])\)\s*', 'letter_upper_bracket'),  # 匹配：(A)(B)等大写字母带英文括号
            (r'^\s*\(([a-z])\)\s*', 'letter_lower_bracket'),  # 匹配：(a)(b)等小写字母带英文括号

            # 罗马数字序号
            (r'^\s*([IVXLCDMivxlcdm]+)\s*[、]\s*', 'roman_dot'),  # 匹配：I、II、i、ii、等罗马数字后跟顿号
            (r'^\s*（([IVXLCDMivxlcdm]+)）\s*', 'roman_paren'),  # 匹配：（I）（II）等罗马数字带中文括号
            (r'^\s*\(([IVXLCDMivxlcdm]+)\)\s*', 'roman_bracket'),  # 匹配：(I)(II)等罗马数字带英文括号

            # 特殊符号序号
            (r'^\s*([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*', 'circle_number'),  # 匹配：①②③等带圈数字
            (r'^\s*([⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇])\s*', 'paren_number'),  # 匹配：⑴⑵⑶等带括号数字
            (r'^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ])\s*', 'roman_special_upper'),  # 匹配：ⅠⅡⅢⅣⅤ等特殊大写罗马数字
            (r'^\s*([ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻ])\s*', 'roman_special_lower'),  # 匹配：ⅰⅱⅲⅳⅴ等特殊小写罗马数字

            # 章节标题
            (r'^\s*(第[一二三四五六七八九十\d]+章)\s*', 'chapter'),  # 匹配：第一章、第1章等章标题
            (r'^\s*(第[一二三四五六七八九十\d]+节)\s*', 'section'),  # 匹配：第一节、第1节等节标题
            (r'^\s*(第[一二三四五六七八九十\d]+条)\s*', 'article'),  # 匹配：第一条、第1条等条标题
            (r'^\s*(第[一二三四五六七八九十\d]+款)\s*', 'clause'),  # 匹配：第一款、第1款等款标题
            (r'^\s*(第[一二三四五六七八九十\d]+项)\s*', 'item'),  # 匹配：第一项、第1项等项标题
        ]

        # 起始序号定义
        self.start_sequences = {
            'chinese_dot': ['一'],
            'chinese_period': ['一'],
            'chinese_paren': ['一'],
            'chinese_bracket': ['一'],
            'arabic_dot': ['1'],
            'arabic_period': ['1'],
            'arabic_right_paren': ['1'],
            'arabic_paren': ['1'],
            'arabic_bracket': ['1'],
            'letter_upper_dot': ['A'],
            'letter_lower_dot': ['a'],
            'letter_upper_paren': ['A'],
            'letter_lower_paren': ['a'],
            'letter_upper_bracket': ['A'],
            'letter_lower_bracket': ['a'],
            'roman_dot': ['I', 'i'],
            'roman_paren': ['I', 'i'],
            'roman_bracket': ['I', 'i'],
            'circle_number': ['①'],
            'paren_number': ['⑴'],
            'roman_special_upper': ['Ⅰ'],
            'roman_special_lower': ['ⅰ'],
            'chapter': [],  # 章节类型没有固定起始
            'section': [],
            'article': [],
            'clause': [],
            'item': [],
        }

    def extract_sequences_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF文件中提取序号信息"""
        sequences = []

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            print(f"开始处理PDF文件: {pdf_path}")
            print(f"总页数: {total_pages}")

            for page_num in range(total_pages):
                page = doc[page_num]
                page_sequences = self._extract_page_sequences(page, page_num + 1)
                sequences.extend(page_sequences)

                if page_sequences:
                    print(f"第 {page_num + 1} 页找到 {len(page_sequences)} 个序号")

            doc.close()

        except Exception as e:
            print(f"处理PDF文件时出错: {str(e)}")
            return []

        return sequences

    def _extract_page_sequences(self, page, page_num: int) -> List[Dict[str, Any]]:
        """从单页中提取序号"""
        sequences = []

        # 获取页面的文本块信息
        text_dict = page.get_text("dict")

        # 检测表格区域
        table_areas = self._detect_table_areas(page)

        # 临时存储同一行的文本片段
        line_fragments = {}  # 格式: {y坐标: [(文本, bbox, 序号信息)]}
        y_tolerance = 1  # y坐标容差值，判断是否为同一行

        # 第一步：收集所有文本片段并按行分组
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # 获取文本的边界框
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    y_pos = round(bbox[1])  # 使用y坐标的整数部分作为行标识

                    # 检查文本是否在表格区域内
                    if self._is_in_table_area(bbox, table_areas):
                        continue  # 跳过表格内的文本

                    # 检查是否为序号
                    sequence_match = self._identify_sequence(text)

                    # 将文本片段添加到对应行
                    found_line = False
                    for key_y in line_fragments.keys():
                        if abs(key_y - y_pos) <= y_tolerance:
                            line_fragments[key_y].append((text, bbox, sequence_match))
                            found_line = True
                            break

                    if not found_line:
                        line_fragments[y_pos] = [(text, bbox, sequence_match)]

        # 第二步：处理每一行的文本片段
        for y_pos, fragments in sorted(line_fragments.items()):
            # 按x坐标排序片段
            fragments.sort(key=lambda x: x[1][0])

            # 合并同一行的文本
            merged_text = "".join(fragment[0] for fragment in fragments)

            # 使用合并后的文本重新检查序号
            sequence_match = self._identify_sequence(merged_text)
            if sequence_match:
                # 计算合并后的边界框
                min_x = min(fragment[1][0] for fragment in fragments)
                min_y = min(fragment[1][1] for fragment in fragments)
                max_x = max(fragment[1][2] for fragment in fragments)
                max_y = max(fragment[1][3] for fragment in fragments)
                merged_bbox = [min_x, min_y, max_x, max_y]

                # 检查序号前面的字符数量是否不超过两个字符
                if self._is_valid_title_position(merged_text, sequence_match[0]):
                    # 检查序号后的内容是否像标题（不是长句子）
                    if self._is_valid_title_content(merged_text, sequence_match[0]):
                        # 检查序号的左边距是否合理（特别是对于1）类型的序号）
                        if self._is_valid_sequence_margin(merged_bbox, sequence_match[1], page):
                            sequence_info = {
                                "text": merged_text,
                                "page": page_num,
                                "rect": merged_bbox,
                                "sequence_type": sequence_match[1],
                                "value": sequence_match[0]
                            }
                            sequences.append(sequence_info)
            # else:
            #     # 如果合并后不是序号，检查单个片段是否为序号
            #     for text, bbox, seq_match in fragments:
            #         if seq_match:
            #             # 检查序号前面的字符数量是否不超过两个字符
            #             if self._is_valid_title_position(text, seq_match[0]):
            #                 sequence_info = {
            #                     "text": text,
            #                     "page": page_num,
            #                     "rect": list(bbox),
            #                     "sequence_type": seq_match[1],
            #                     "value": seq_match[0]
            #                 }
            #                 sequences.append(sequence_info)

        # 按照在页面中的位置排序（从上到下，从左到右）
        sequences.sort(key=lambda x: (x["rect"][1], x["rect"][0]))

        return sequences

    def _identify_sequence(self, text: str) -> Optional[Tuple[str, str]]:
        """识别文本中的序号"""
        text = text.strip()
        if not text:
            return None

        # 排除包含之的正则
        if re.search(r'”之', text):
            return None

        # 排除以金额和百分比开头的文本
        # if re.search(r'^\s*\d+\.?\d*\s*[元万%股]', text):
        #     return None

        # 排除以六个数字开头的文本（证券代码等）
        if re.search(r'^\d{6}', text):
            return None

        # 检查每个序号模式
        for pattern, seq_type in self.patterns:
            match = re.search(pattern, text)
            if match:
                extracted_value = match.group(1)  # 提取的数字/字母部分
                full_sequence = match.group(0).strip()  # 完整的序号（包括括号等）

                # 检查编号数值是否超过100
                if self._is_valid_sequence_number(extracted_value):
                    return full_sequence, seq_type

        return None

    def _is_valid_title_position(self, text: str, sequence_value: str) -> bool:
        """检查序号前面的字符数量是否不超过两个字符"""
        # 在文本中找到序号的位置
        sequence_pos = text.find(sequence_value)
        if sequence_pos == -1:
            # 如果找不到序号，可能是因为序号包含特殊字符，尝试更宽松的匹配
            # 去掉序号中的特殊字符进行匹配
            clean_sequence = sequence_value.strip('（）()、. ')
            for i, char in enumerate(text):
                if clean_sequence.startswith(char):
                    # 找到可能的起始位置，检查是否匹配
                    potential_match = text[i:i + len(clean_sequence)]
                    if clean_sequence in potential_match:
                        sequence_pos = i
                        break

        if sequence_pos == -1:
            # 仍然找不到，返回False（过滤掉）
            return False

        # 计算序号前面的字符数量
        prefix_text = text[:sequence_pos].strip()

        # 过滤掉一些常见的非内容字符（空格、标点等）
        # 只计算实际的文字字符
        content_chars = 0
        for char in prefix_text:
            if char.isalnum() or '\u4e00' <= char <= '\u9fff':  # 字母、数字或中文字符
                content_chars += 1

        # 序号前面不能超过两个字符
        return content_chars <= 2

    def _is_valid_title_content(self, text: str, sequence_value: str) -> bool:
        """检查序号后的内容是否像标题（不是长句子）"""
        # 找到序号的位置
        sequence_pos = text.find(sequence_value)
        if sequence_pos == -1:
            return False

        # 获取序号后的内容
        content_after_sequence = text[sequence_pos + len(sequence_value):].strip()

        # 如果序号后没有内容，过滤掉
        if not content_after_sequence:
            return False

        # 检查内容长度，过长的可能不是标题
        if len(content_after_sequence) > 200:
            return False

        # 检查是否包含过多的句号，可能是长句子而不是标题
        period_count = content_after_sequence.count('。') + content_after_sequence.count('.')
        if period_count > 2:
            return False

        return True

    def _is_valid_sequence_margin(self, bbox: List[float], seq_type: str, page) -> bool:
        """检查序号的左边距是否合理（特别是对于1）类型的序号）"""
        # 对于 arabic_right_paren 类型（1）格式），需要检查左边距
        if seq_type == 'arabic_right_paren':
            # 获取页面中所有文本的左边距，找到正文开始位置
            text_left_margins = self._get_page_text_left_margins(page)

            if not text_left_margins:
                return True  # 如果无法获取文本边距信息，默认通过

            # 找到最常见的左边距位置（正文开始位置）
            main_text_left_margin = self._find_main_text_left_margin(text_left_margins)

            # 序号的左边距
            sequence_left_margin = bbox[0]

            # 如果序号的左边距明显小于正文左边距，说明是直接顶行的序号，需要过滤
            # 允许一定的容差（比如10个像素）
            margin_tolerance = 5

            if sequence_left_margin < main_text_left_margin + margin_tolerance:
                return False

        return True

    def _get_page_text_left_margins(self, page) -> List[float]:
        """获取页面中所有文本的左边距"""
        left_margins = []

        try:
            # 获取页面的文本块信息
            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        # 获取文本的边界框
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        left_margin = bbox[0]

                        # 过滤掉明显异常的边距值
                        if left_margin > 0:
                            left_margins.append(left_margin)

        except Exception:
            pass

        return left_margins

    def _find_main_text_left_margin(self, left_margins: List[float]) -> float:
        """找到最靠左的正文位置（正文开始位置）"""
        if not left_margins:
            return 0

        # 对左边距进行分组，容差为5像素
        margin_tolerance = 5
        margin_groups = {}

        for margin in left_margins:
            # 找到最接近的分组
            found_group = False
            for group_key in margin_groups.keys():
                if abs(margin - group_key) <= margin_tolerance:
                    margin_groups[group_key].append(margin)
                    found_group = True
                    break

            if not found_group:
                margin_groups[margin] = [margin]

        # 找到最靠左的分组（最小的左边距）
        # 但要求该分组至少有一定数量的文本，避免异常值
        min_group_size = max(1, len(left_margins) // 20)  # 至少要有总数的5%

        valid_groups = [(group_key, group_margins) for group_key, group_margins in margin_groups.items()
                        if len(group_margins) >= min_group_size]

        if not valid_groups:
            # 如果没有足够大的分组，就取所有分组中最小的
            min_margin = min(margin_groups.keys())
            return min_margin

        # 在有效分组中找到最靠左的
        min_group_key = min(valid_groups, key=lambda x: x[0])[0]
        min_group_margins = margin_groups[min_group_key]

        # 使用该分组的平均值作为主要左边距
        main_margin = sum(min_group_margins) / len(min_group_margins)

        return main_margin

    def _is_valid_sequence_number(self, value: str) -> bool:
        """检查序号数值是否有效（不超过100）"""
        # 转换为数字进行检查
        num = self._chinese_to_number(value)
        if num is not None:
            return num <= 100

        # 对于无法转换的值（如罗马数字、字母等），暂时允许通过
        # 可以根据需要添加更多的检查逻辑
        return True

    def _is_start_sequence(self, seq_type: str, value: str) -> bool:
        """判断是否为起始序号"""
        start_values = self.start_sequences.get(seq_type, [])
        # 从完整序号中提取数字/字母部分进行比较
        extracted_value = self._extract_sequence_value(value, seq_type)
        return extracted_value in start_values

    def build_tree(self, sequences: List[Dict[str, Any]]) -> List[SequenceNode]:
        """构建序号树形结构"""
        if not sequences:
            return []

        root_nodes = []
        stack = []  # 用于跟踪当前层级的节点栈

        for seq_info in sequences:
            seq_type = seq_info["sequence_type"]
            value = seq_info["value"]

            # 移除几种特殊情况前的引号和注
            seq_info["text"] = seq_info["text"].lstrip("‘’“”\"\'")

            # 创建新节点
            node = SequenceNode(
                text=seq_info["text"],
                page=seq_info["page"],
                rect=seq_info["rect"],
                sequence_type=seq_type,
                value=value,
                level=0,
                children=[]
            )

            if not stack:
                # 第一个节点，作为根节点
                node.level = 1
                root_nodes.append(node)
                stack.append(node)
            else:
                last_node = stack[-1]

                if self._is_same_level_sequence(last_node.sequence_type, last_node.text, last_node.rect, seq_type,
                                                seq_info["text"], seq_info["rect"]):
                    # 同类型序号且确实是同级关系
                    node.level = last_node.level
                    stack.pop()  # 移除上一个同级节点

                    if len(stack) > 0:
                        # 添加到父节点的子节点列表
                        stack[-1].children.append(node)
                    else:
                        # 根级节点
                        root_nodes.append(node)

                    stack.append(node)

                elif self._is_start_sequence(seq_type, value):
                    # 新的起始序号，作为上一个节点的子节点
                    node.level = last_node.level + 1
                    last_node.children.append(node)
                    stack.append(node)

                else:
                    # 不是起始序号，需要返回到合适的层级
                    # 首先检查是否与栈中某个节点同类型且同层级（同级关系）
                    found_sibling = False
                    for i in range(len(stack) - 1, -1, -1):
                        if self._is_same_level_sequence(stack[i].sequence_type, stack[i].text, stack[i].rect, seq_type,
                                                        seq_info["text"], seq_info["rect"]):
                            # 找到同类型且同层级的节点，作为同级关系
                            node.level = stack[i].level
                            # 回退到该节点的父级
                            if i > 0:
                                # 有父节点，添加到父节点的子节点列表
                                parent = stack[i - 1]
                                parent.children.append(node)
                                # 更新栈，保留到父节点，然后添加当前节点
                                stack = stack[:i] + [node]
                            else:
                                # 没有父节点，作为根节点
                                root_nodes.append(node)
                                stack = [node]
                            found_sibling = True
                            break

                    if not found_sibling:
                        # 没找到同类型，向上查找合适的父节点
                        while len(stack) > 1:
                            stack.pop()
                            if len(stack) > 0:
                                parent = stack[-1]
                                # 检查是否可以作为同级节点
                                if self._can_be_sibling(parent.sequence_type, seq_type):
                                    node.level = parent.level
                                    stack.pop()
                                    if len(stack) > 0:
                                        stack[-1].children.append(node)
                                    else:
                                        root_nodes.append(node)
                                    stack.append(node)
                                    break
                        else:
                            # 如果没有找到合适的父节点，作为根节点
                            node.level = 1
                            root_nodes.append(node)
                            stack = [node]

        return root_nodes

    def _can_be_sibling(self, parent_type: str, current_type: str) -> bool:
        """判断两个序号类型是否可以是同级关系"""
        # 简化的同级判断逻辑，可以根据需要扩展
        type_groups = [
            ['chinese_dot', 'chinese_paren', 'chinese_bracket'],
            ['arabic_dot', 'arabic_paren', 'arabic_bracket', 'arabic_right_paren', 'arabic_square'],
            ['letter_upper_dot', 'letter_upper_paren', 'letter_upper_bracket'],
            ['letter_lower_dot', 'letter_lower_paren', 'letter_lower_bracket'],
            ['roman_dot', 'roman_paren', 'roman_bracket'],
            ['circle_number', 'paren_number'],
            ['roman_special_upper', 'roman_special_lower'],
            ['multi_level_2', 'multi_level_3', 'multi_level_4'],
            ['chapter', 'section', 'article', 'clause', 'item']
        ]

        for group in type_groups:
            if parent_type in group and current_type in group:
                return True

        return False

    def _is_same_level_sequence(self, seq_type1: str, text1: str, rect1: List[float], seq_type2: str, text2: str,
                                rect2: List[float]) -> bool:
        """判断两个序号是否属于同一层级（包含类型判断、连续性判断和位置判断）"""
        # 首先判断类型是否相同
        if seq_type1 != seq_type2:
            return False

        # 检查x坐标位置是否相近（同级别的序号应该有相同的左边距）
        # 特别针对汉字数字类型进行位置检查
        if seq_type1 in ['chinese_dot', 'chinese_period', 'chinese_paren', 'chinese_bracket']:
            x1 = rect1[0]  # 第一个序号的左边距
            x2 = rect2[0]  # 第二个序号的左边距
            position_tolerance = 10  # 位置容差，10像素

            if abs(x1 - x2) > position_tolerance:
                return False  # 位置差异太大，不是同级

        seq_type = seq_type1  # 类型相同，使用其中一个

        # 对于带括号的序号（中文括号和阿拉伯数字括号），需要判断序号值的连续性
        if seq_type in ['chinese_paren', 'arabic_paren', 'chinese_bracket', 'arabic_bracket', 'arabic_right_paren']:
            # 提取序号值
            value1 = self._extract_sequence_value(text1, seq_type)
            value2 = self._extract_sequence_value(text2, seq_type)

            if value1 and value2:
                # 转换为数字进行比较
                num1 = self._chinese_to_number(value1)
                num2 = self._chinese_to_number(value2)

                if num1 and num2:
                    # 基于连续性和数值范围的判断逻辑
                    diff = abs(num1 - num2)

                    # 连续性判断：相差1认为是连续的同级序号
                    if diff == 1:
                        return True

                    # 对于相差较大的情况，基于数值范围判断
                    elif diff > 1:
                        # 如果两个数字都>3，可能是主项级别的同级（如（4）和（5））
                        if num1 > 3 and num2 > 3:
                            return True
                        # 如果两个数字都≤3，很可能是子项级别的跳跃，不认为同级
                        elif num1 <= 3 and num2 <= 3:
                            return False
                        # 如果一个≤3一个>3，很可能是跨层级，不认为同级
                        else:
                            return False

        # 对于其他类型的序号，也进行连续性判断
        elif seq_type in ['arabic_dot', 'arabic_period', 'chinese_dot', 'chinese_period']:
            value1 = self._extract_sequence_value(text1, seq_type)
            value2 = self._extract_sequence_value(text2, seq_type)

            if value1 and value2:
                num1 = self._chinese_to_number(value1)
                num2 = self._chinese_to_number(value2)

                if num1 and num2:
                    # 对于顿号和句号类型，连续性判断更严格
                    diff = abs(num1 - num2)
                    return diff <= 2

        # 对于其他类型，保持原有逻辑
        return True

    def _extract_sequence_value(self, text: str, seq_type: str) -> Optional[str]:
        """从文本中提取序号值（数字/字母部分，用于连续性判断）"""
        for pattern, pattern_type in self.patterns:
            if pattern_type == seq_type:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
        return None

    def _chinese_to_number(self, chinese_num: str) -> Optional[int]:
        """将中文数字转换为阿拉伯数字"""
        # 如果是阿拉伯数字，直接转换
        if chinese_num.isdigit():
            try:
                return int(chinese_num)
            except ValueError:
                pass

        # 扩展的数字映射，包括中文数字和特殊符号
        number_map = {
            # 中文数字
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '零': 0,

            # 带圈数字
            '①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5,
            '⑥': 6, '⑦': 7, '⑧': 8, '⑨': 9, '⑩': 10,
            '⑪': 11, '⑫': 12, '⑬': 13, '⑭': 14, '⑮': 15,
            '⑯': 16, '⑰': 17, '⑱': 18, '⑲': 19, '⑳': 20,

            # 带括号数字
            '⑴': 1, '⑵': 2, '⑶': 3, '⑷': 4, '⑸': 5,
            '⑹': 6, '⑺': 7, '⑻': 8, '⑼': 9, '⑽': 10,
            '⑾': 11, '⑿': 12, '⒀': 13, '⒁': 14, '⒂': 15,
            '⒃': 16, '⒄': 17, '⒅': 18, '⒆': 19, '⒇': 20,

            # 罗马数字（大写）
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
            'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
            'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
            'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,

            # 罗马数字（小写）
            'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
            'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
            'xi': 11, 'xii': 12, 'xiii': 13, 'xiv': 14, 'xv': 15,
            'xvi': 16, 'xvii': 17, 'xviii': 18, 'xix': 19, 'xx': 20,

            # 特殊罗马数字
            'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4, 'Ⅴ': 5,
            'Ⅵ': 6, 'Ⅶ': 7, 'Ⅷ': 8, 'Ⅸ': 9, 'Ⅹ': 10,
            'Ⅺ': 11, 'Ⅻ': 12,

            'ⅰ': 1, 'ⅱ': 2, 'ⅲ': 3, 'ⅳ': 4, 'ⅴ': 5,
            'ⅵ': 6, 'ⅶ': 7, 'ⅷ': 8, 'ⅸ': 9, 'ⅹ': 10,
            'ⅺ': 11, 'ⅻ': 12,
        }

        # 简单的单个数字/符号
        if chinese_num in number_map:
            return number_map[chinese_num]

        # 处理复合中文数字（如二十、三十、九十九等）
        if '十' in chinese_num:
            try:
                if chinese_num == '十':
                    return 10
                elif chinese_num.startswith('十'):
                    # 十一、十二等
                    unit = chinese_num[1:]
                    if unit in number_map:
                        return 10 + number_map[unit]
                elif chinese_num.endswith('十'):
                    # 二十、三十等
                    tens = chinese_num[0]
                    if tens in number_map:
                        return number_map[tens] * 10
                else:
                    # 二十一、三十五等
                    parts = chinese_num.split('十')
                    if len(parts) == 2 and parts[0] in number_map and parts[1] in number_map:
                        return number_map[parts[0]] * 10 + number_map[parts[1]]
            except:
                pass

        # 处理字母序号
        if len(chinese_num) == 1:
            if 'A' <= chinese_num <= 'Z':
                return ord(chinese_num) - ord('A') + 1
            elif 'a' <= chinese_num <= 'z':
                return ord(chinese_num) - ord('a') + 1

        return None

    def _detect_table_areas(self, page) -> List[List[float]]:
        """检测页面中的表格区域"""
        table_areas = []

        try:
            # 方法1：使用PyMuPDF的表格检测功能，但需要验证
            tables = page.find_tables()
            for table in tables:
                # 验证这个区域是否真的是表格
                if self._validate_table_area(page, table):
                    bbox = table.bbox
                    table_areas.append(list(bbox))
        except:
            # 如果表格检测失败，使用备用方法
            pass

        return table_areas

    def _validate_table_area(self, page, table) -> bool:
        """验证检测到的区域是否真的是表格 - 过滤表头有多列但只有第一列有文字且文字较长的情况"""
        try:
            # 检查table对象是否有header属性
            if not hasattr(table, 'header') or not table.header:
                return True  # 没有表头信息时，保守地认为是表格

            header = table.header
            header_names = header.names

            # 检查是否只有第一列有文字且文字较长的情况
            # 统计有内容的列数
            non_empty_columns = []
            for i, name in enumerate(header_names):
                if name and name.strip():
                    non_empty_columns.append(i)

            # 如果只有第一列有内容
            if len(non_empty_columns) == 1 and non_empty_columns[0] == 0:
                first_column_text = header_names[0].strip()
                first_column_length = len(first_column_text)

                # 如果第一列文字比较长（超过20个字符），认为不是真正的表格
                if first_column_length > 20:
                    return False

            # 其他情况都认为是有效表格
            return True

        except:
            # 如果验证失败，保守地认为是表格
            return True

    def _is_in_table_area(self, bbox: List[float], table_areas: List[List[float]]) -> bool:
        """检查文本边界框是否在表格区域内"""
        if not table_areas:
            return False

        text_x1, text_y1, text_x2, text_y2 = bbox

        for table_bbox in table_areas:
            table_x1, table_y1, table_x2, table_y2 = table_bbox

            # 检查文本是否与表格区域重叠
            if (text_x1 < table_x2 and text_x2 > table_x1 and
                    text_y1 < table_y2 and text_y2 > table_y1):
                return True

        return False

    def check_sequence_errors(self, tree_nodes: List[SequenceNode]) -> List[Dict[str, Any]]:
        """检查序号错误（重复、缺失、顺序错误等）"""
        errors = []

        def check_node_sequence(nodes: List[SequenceNode], parent_info: str = ""):
            """递归检查节点序号"""
            if not nodes:
                return

            # 按序号类型分组
            type_groups = {}
            for node in nodes:
                seq_type = node.sequence_type
                if seq_type not in type_groups:
                    type_groups[seq_type] = []
                type_groups[seq_type].append(node)

            # 检查每个类型的序号
            for seq_type, type_nodes in type_groups.items():
                if len(type_nodes) <= 1:
                    # 单个节点，递归检查子节点
                    for node in type_nodes:
                        check_node_sequence(node.children, f"{parent_info}{node.value}")
                    continue

                # 提取序号数值并排序
                node_values = []
                for node in type_nodes:
                    extracted_value = self._extract_sequence_value(node.value, seq_type)
                    if extracted_value:
                        num_value = self._chinese_to_number(extracted_value)
                        if num_value is not None:
                            node_values.append((num_value, node, extracted_value))

                if not node_values:
                    # 无法提取数值，跳过检查，但仍检查子节点
                    for node in type_nodes:
                        check_node_sequence(node.children, f"{parent_info}{node.value}")
                    continue

                # 按数值排序
                node_values.sort(key=lambda x: x[0])

                # 检查序号错误
                prev_num = None
                prev_node = None

                for i, (num_value, node, extracted_value) in enumerate(node_values):
                    current_info = f"{parent_info}{node.value}"

                    if prev_num is not None:
                        # 检查重复
                        if num_value == prev_num:
                            errors.append({
                                "error_type": "duplicate",
                                "error_sequence": node.value,
                                "page": node.page,
                                "rect": node.rect,
                                "previous_sequence": prev_node.value if prev_node else None,
                                "description": f"序号重复，与前一个序号相同：{prev_node.value if prev_node else '未知'}"
                            })

                        # 检查缺失（跳跃）
                        elif num_value > prev_num + 1:
                            missing_nums = list(range(prev_num + 1, num_value))
                            for missing_num in missing_nums:
                                missing_value = self._number_to_sequence_format(missing_num, seq_type, extracted_value)
                                errors.append({
                                    "error_type": "missing",
                                    "error_sequence": missing_value,
                                    "page": node.page,
                                    "rect": node.rect,
                                    "previous_sequence": prev_node.value if prev_node else None,
                                    "description": f"序号缺失：{missing_value}，上一个序号为：{prev_node.value if prev_node else '未知'}"
                                })

                        # 检查顺序错误（数值较小但位置在后）
                        elif num_value < prev_num:
                            errors.append({
                                "error_type": "order",
                                "error_sequence": node.value,
                                "page": node.page,
                                "rect": node.rect,
                                "previous_sequence": prev_node.value if prev_node else None,
                                "description": f"序号顺序错误，请核对，上一个序号为：{prev_node.value if prev_node else '未知'}"
                            })

                    prev_num = num_value
                    prev_node = node

                    # 递归检查子节点
                    check_node_sequence(node.children, current_info)

        # 开始检查
        check_node_sequence(tree_nodes)
        return errors

    def _number_to_sequence_format(self, num: int, seq_type: str, sample_value: str) -> str:
        """将数字转换为对应的序号格式"""
        # 根据序号类型和样本值生成对应格式的序号
        if seq_type == 'chinese_paren':
            chinese_num = self._number_to_chinese(num)
            return f"（{chinese_num}）" if chinese_num else f"（{num}）"
        elif seq_type == 'arabic_paren':
            return f"（{num}）"
        elif seq_type == 'arabic_right_paren':
            return f"{num}）"
        elif seq_type == 'chinese_bracket':
            chinese_num = self._number_to_chinese(num)
            return f"({chinese_num})" if chinese_num else f"({num})"
        elif seq_type == 'arabic_bracket':
            return f"({num})"
        elif seq_type == 'chinese_dot':
            chinese_num = self._number_to_chinese(num)
            return f"{chinese_num}、" if chinese_num else f"{num}、"
        elif seq_type == 'arabic_dot':
            return f"{num}、"
        elif seq_type == 'chinese_period':
            chinese_num = self._number_to_chinese(num)
            return f"{chinese_num}." if chinese_num else f"{num}."
        elif seq_type == 'arabic_period':
            return f"{num}."
        elif seq_type in ['letter_upper_dot', 'letter_upper_paren', 'letter_upper_bracket']:
            if num <= 26:
                letter = chr(ord('A') + num - 1)
                if seq_type == 'letter_upper_dot':
                    return f"{letter}、"
                elif seq_type == 'letter_upper_paren':
                    return f"（{letter}）"
                else:  # letter_upper_bracket
                    return f"({letter})"
        elif seq_type in ['letter_lower_dot', 'letter_lower_paren', 'letter_lower_bracket']:
            if num <= 26:
                letter = chr(ord('a') + num - 1)
                if seq_type == 'letter_lower_dot':
                    return f"{letter}、"
                elif seq_type == 'letter_lower_paren':
                    return f"（{letter}）"
                else:  # letter_lower_bracket
                    return f"({letter})"

        # 默认返回数字格式
        return str(num)

    def _number_to_chinese(self, num: int) -> Optional[str]:
        """将数字转换为中文数字"""
        if num < 1 or num > 99:
            return None

        chinese_digits = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']

        if num <= 10:
            if num == 10:
                return '十'
            return chinese_digits[num]
        elif num < 20:
            return f"十{chinese_digits[num - 10]}"
        else:
            tens = num // 10
            units = num % 10
            if units == 0:
                return f"{chinese_digits[tens]}十"
            else:
                return f"{chinese_digits[tens]}十{chinese_digits[units]}"

    def print_tree(self, nodes: List[SequenceNode], indent: str = "") -> str:
        """打印树形结构"""
        result = ""
        for node in nodes:
            result += f"{indent}{node.text}\n"
            if node.children:
                result += self.print_tree(node.children, indent + "    ")
        return result

    def save_results(self, sequences: List[Dict[str, Any]], tree_nodes: List[SequenceNode],
                     output_dir: str = ""):
        """保存结果到文件"""
        import os

        # 保存原始序号数据
        sequences_file = os.path.join(output_dir, "sequences.json")
        with open(sequences_file, 'w', encoding='utf-8') as f:
            json.dump(sequences, f, ensure_ascii=False, indent=2)

        # 保存树形结构数据
        tree_data = [node.to_dict() for node in tree_nodes]
        tree_file = os.path.join(output_dir, "sequence_tree.json")
        with open(tree_file, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)

        # 保存树形结构文本
        tree_text = self.print_tree(tree_nodes)
        text_file = os.path.join(output_dir, "sequence_tree.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(tree_text)

        # 检查序号错误并保存
        errors = self.check_sequence_errors(tree_nodes)
        errors_file = os.path.join(output_dir, "sequence_errors.json")
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到:")
        print(f"  - 原始数据: {sequences_file}")
        print(f"  - 树形数据: {tree_file}")
        print(f"  - 树形文本: {text_file}")
        print(f"  - 错误检查: {errors_file}")

        if errors:
            print(f"\n发现 {len(errors)} 个序号错误:")
            for error in errors:
                print(f"  - {error['error_type']}: {error['description']}")
        else:
            print("\n未发现序号错误")

    def process_pdf(self, pdf_path: str, output_dir: str = ""):
        """处理PDF文件的完整流程"""
        print("开始处理PDF文件...")

        # 提取序号
        sequences = self.extract_sequences_from_pdf(pdf_path)
        if not sequences:
            print("未找到任何序号")
            return

        print(f"总共找到 {len(sequences)} 个序号")

        # 构建树形结构
        tree_nodes = self.build_tree(sequences)

        # 打印树形结构
        print("\n序号树形结构:")
        print(self.print_tree(tree_nodes))

        # 保存结果
        self.save_results(sequences, tree_nodes, output_dir)


if __name__ == "__main__":
    # 创建处理器
    builder = PDFSequenceTreeBuilder()

    # 处理PDF文件
    pdf_path = "序号测试3.pdf"
    builder.process_pdf(pdf_path)
