# !/usr/bin/env python3


import os
from typing import List

import torch
from transformers import AutoTokenizer

from model import convert_inputs, get_bool_ids_greater_than, get_span


def inference(
    model,
    tokenizer,
    device: str,
    contents: List[str], 
    prompts: List[str], 
    max_length=512, 
    prob_threshold=0.5
    ) -> List[str]:
    """
    输入 promot 和 content 列表，返回模型提取结果。    

    Args:
        contents (List[str]): 待提取文本列表, e.g. -> [
                                                    '《琅琊榜》是胡歌主演的一部电视剧。',
                                                    '《笑傲江湖》是一部金庸的著名小说。',
                                                    ...
                                                ]
        prompts (List[str]): prompt列表，用于告知模型提取内容, e.g. -> [
                                                                    '主语',
                                                                    '类型',
                                                                    ...
                                                                ]
        max_length (int): 句子最大长度，小于最大长度则padding，大于最大长度则截断。
        prob_threshold (float): sigmoid概率阈值，大于该阈值则二值化为True。

    Returns:
        List: 模型识别结果, e.g. -> [['琅琊榜'], ['电视剧']]
    """
    inputs = convert_inputs(tokenizer, prompts, contents, max_length=max_length)
    model_inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'token_type_ids': inputs['token_type_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
    }
    output_sp, output_ep = model(**model_inputs)
    output_sp, output_ep = output_sp.detach().cpu().tolist(), output_ep.detach().cpu().tolist()
    start_ids_list = get_bool_ids_greater_than(output_sp, prob_threshold)
    end_ids_list = get_bool_ids_greater_than(output_ep, prob_threshold)

    res = []                                                    # decode模型输出，将token id转换为span text
    offset_mapping = inputs['offset_mapping'].tolist()
    for start_ids, end_ids, prompt, content, offset_map in zip(start_ids_list, 
                                                            end_ids_list,
                                                            prompts,
                                                            contents,
                                                            offset_mapping):
        span_set = get_span(start_ids, end_ids)                 # e.g. {(5, 7), (9, 10)}
        current_span_list = []
        for span in span_set:
            if span[0] < len(prompt) + 2:                       # 若答案出现在promot区域，过滤
                continue
            span_text = ''                                      # 答案span
            input_content = prompt + content                    # 对齐token_ids
            for s in range(span[0], span[1] + 1):               # 将 offset map 里 token 对应的文本切回来
                span_text += input_content[offset_map[s][0]: offset_map[s][1]]  #[ ]既然只用了offset_map的后半部分，那一开始就只保存后办法也行吧
            current_span_list.append(span_text)
        res.append(current_span_list)
    return res


def event_extract_example(
    model,
    tokenizer,
    device: str,
    sentence: str, 
    schema: dict, 
    prob_threshold=0.6,
    max_seq_len=128,
    ) -> dict:
    """
    UIE事件抽取示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '5月17号晚上10点35分加班打车回家，36块五。'
        schema (dict): 事件定义字典, e.g. -> {
                                            '加班触发词': ['时间','地点'],
                                            '出行触发词': ['时间', '出发地', '目的地', '花费']
                                        }
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。
    
    Returns:
        dict -> {
                '触发词1': {},
                '触发词2': {
                    '事件属性1': [属性值1, 属性值2, ...],
                    '事件属性2': [属性值1, 属性值2, ...],
                    '事件属性3': [属性值1, 属性值2, ...],
                    ...
                }
            }
    """
    rsp = {}
    trigger_prompts = list(schema.keys())

    for trigger_prompt in trigger_prompts:
        rsp[trigger_prompt] = {}
        triggers = inference(
            model,
            tokenizer,
            device,
            [sentence], 
            [trigger_prompt], 
            max_length=128, 
            prob_threshold=prob_threshold)[0]
        
        for trigger in triggers:
            if trigger:
                arguments = schema.get(trigger_prompt)
                contents = [sentence] * len(arguments)
                prompts = [f"{trigger}的{a}" for a in arguments]
                res = inference(
                    model,
                    tokenizer,
                    device,
                    contents, 
                    prompts,
                    max_length=max_seq_len, 
                    prob_threshold=prob_threshold)
                for a, r in zip(arguments, res):
                    rsp[trigger_prompt][a] = r
    print('[+] Event-Extraction Results: ', rsp)


def information_extract_example(
    model,
    tokenizer,
    device: str,
    sentence: str, 
    schema: dict, 
    prob_threshold=0.6, 
    max_seq_len=128
    ) -> dict:
    """
    UIE信息抽取示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '麻雀是几级保护动物？国家二级保护动物'
        schema (dict): 事件定义字典, e.g. -> {
                                            '主语': ['保护等级']
                                        }
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。
    
    Returns:
        dict -> {
                '麻雀': {
                        '保护等级': ['国家二级']
                    },
                ...
            }
    """
    rsp = {}
    subject_prompts = list(schema.keys())

    for subject_prompt in subject_prompts:  #对一个句子，抽取每一个schema_dict key
        subjects = inference(
            model,
            tokenizer,
            device,
            [sentence], 
            [subject_prompt], 
            max_length=128, 
            prob_threshold=prob_threshold)[0]
        
        for subject in subjects:  #对一个句子抽出的schema_dict key结果为subject，对每一个subject
            if subject:
                rsp[subject] = {}
                predicates = schema.get(subject_prompt)
                contents = [sentence] * len(predicates)
                prompts = [f"{subject}的{p}" for p in predicates]  #拼接subject和相关所有属性，做类似ner的span抽取
                res = inference(
                    model,
                    tokenizer,
                    device,
                    contents, 
                    prompts,
                    max_length=max_seq_len, 
                    prob_threshold=prob_threshold
                )
                for p, r in zip(predicates, res):
                    rsp[subject][p] = r
    print('[+] Information-Extraction Results: ', rsp)


def ner_example(
    model,
    tokenizer,
    device: str,
    sentence: str, 
    schema: list, 
    prob_threshold=0.3
    ) -> dict:
    """
    UIE做NER任务示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '5月17号晚上10点35分加班打车回家，36块五。'
        schema (list): 待抽取的实体列表, e.g. -> ['出发地', '目的地', '时间']
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。
    
    Returns:
        dict -> {
                实体1: [实体值1, 实体值2, 实体值3...],
                实体2: [实体值1, 实体值2, 实体值3...],
                ...
            }
    """
    rsp = {}
    sentences = [sentence] * len(schema)    #  一个prompt需要对应一个句子，所以要复制n遍句子
    res = inference(
        model,
        tokenizer,
        device,
        sentences, 
        schema, 
        max_length=128, 
        prob_threshold=prob_threshold)
    for s, r in zip(schema, res):
        rsp[s] = r
    print('[+] NER Results: ', rsp)


if __name__ == "__main__":
    from rich import print

    device = 'cuda:0'                                    
    # saved_model_path = './checkpoints/DuIE/model_best/'     # 训练模型存放地址
    saved_model_path = './uie-base-zh/'
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    # model = torch.load(os.path.join(saved_model_path, 'model.pt'))
    model = torch.load(os.path.join(saved_model_path, 'pytorch_model.bin'))
    model.to(device).eval()

    sentences = [
        # '九龙坡区的数字经济在重庆市处于什么位置？',
        # '创域企业管理顾问有限公司成立于2002年，以帮助中小型企业领导人建立和完善企业管理系统为使命，从而使自己企业成为一个自动运作的商业盈利机构',
        # '请你推荐介绍一下火石创造创始人金霞，以及她靠谱吗？今天她受邀为九龙坡区领导干部授课，讲关于数字时代的九龙坡机会，你为此情此景写首诗表达对九龙坡区春奎书记的敬佩和感谢。',
        # '张衡医生今天刚完成一项手术，该手术比较复杂，涉及手术刀，呼吸机，据说手术中用到的抗氧化皮质手套是智新公司最新上市的新款手套',
        # "叶海波，男，1977年出生于湖北红安县，1997年至2007年就读于武汉大学，分别获得法学学士（2001）、法学硕士（2004）和法学博士（2007）学位，主要研究宪法学与行政法学",
        # '5月17号晚上10点35分加班打车回家，36块五。',
        # "张三于1994年出生于浙江省杭州市",
        '5约13日，杭州市委书记吴越带队视察了百度科技有限公司',
        # "湖南海利生物科技股份有限公司在医药行业中的市场份额如何？",
        # "唐山港在港口物流领域的技术创新和社会责任实践如何？",
        # "三一重工在工程机械行业的技术创新和市场表现如何？",
        # "恒生电子在金融科技领域的技术创新和市场表现如何？",

        # "河南亿阳信通科技股份有限公司在河南省的市场表现如何？",
        # "广东省珠海市的智能制造产业发展如何？",
        # "贵州省黔南州的特色种植业发展如何？有哪些特色作物？",
        # "广西壮族自治区南宁市的特色旅游产业有哪些新的发展方向？",

        # "作为智能家居市场的领军企业，飞利浦公司的智能家居产品是否能够实现更智能化、更便捷化的服务？",
        # "随着时尚行业的发展，Gucci公司是否能够推出更具有创新性和吸引力的产品，以保持品牌的竞争优势？",
        # "随着汽车电动化的发展，特斯拉公司是否能够在全球范围内推出更多的电动车型，并为消费者提供更具吸引力的产品？",
        # "随着越来越多的人开始使用电子支付服务，PayPal公司是否能够在全球范围内继续扩大市场份额？",
        # "中国电信在普及5G网络方面的进展如何？",
        # "中国移动在发展物联网方面取得了哪些成就？",
        # "中国平安在金融科技领域的创新实践如何？",
        # "中国航空工业集团在全球航空领域的技术实力如何？",
        # "招商银行在数字化转型方面的实践和成效如何？",
        # "青岛啤酒在中国啤酒市场的品牌影响力和销售业绩如何？",
        # "建设银行在金融行业的数字化转型和服务质量如何？",
        # "江西赣锋锂业股份有限公司在新能源材料行业中的市场竞争优势如何？",
        # "江苏紫金矿业集团有限公司在金矿开采领域中的技术和资源优势如何？",


    ]
    
    # NER 示例
    # for sentence in sentences:
    #     ner_example(
    #         model,
    #         tokenizer,
    #         device,
    #         sentence=sentence, 
    #         #schema=['机构', '学校', '企业', '地名', '城市', '公司', '产业', '时间', '人名', '产品名', '器械', '学位', '专业']
    #         schema = ['企业', '机构', '人物', '行业领域', '产业领域', '产品名称', '产品', '产业业务', '业务', '区域', '地名']
    #     )

    # SPO抽取示例
    for sentence in sentences:
        information_extract_example(
            model,
            tokenizer,
            device,
            sentence=sentence, 
            schema={
                    # '出行触发词': ['时间', '出发地', '目的地', '花费'],
                    # '加班触发词': ['时间','地点'],
                    '参观、游览、考察触发词':['时间', '地点', '人物'],
                
                }
        )

    # event抽取示例
    for sentence in sentences:
        event_extract_example(
            model,
            tokenizer,
            device,
            sentence=sentence, 
            schema={
                    
                    '参观、游览、考察触发词':['时间', '地点', '人物'],
                
                }
        )


    