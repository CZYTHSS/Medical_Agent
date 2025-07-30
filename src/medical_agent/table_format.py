import pandas as pd

# 保持向后兼容的ROW_INDEX（作为兜底）
ROW_INDEX = {
    '左主干近段(pLM)': 0,
    '左主干中段(mLM)': 1,
    '左主干远段(dLM)': 2,
    '左前降支近段(pLAD)': 3,
    '左前降支中段(mLAD)': 4,
    '左前降支远段(dLAD)': 5,
    '左回旋支近段(pLCX)': 6,
    '左回旋支中段(mLCX)': 7,
    '左回旋支远段(dLCX)': 8,
    '右冠近段(pRCA)': 9,
    '右冠中段(mRCA)': 10,
    '右冠远段(dRCA)': 11,
    '第一对角支(D1)': 12,
    '第二对角支(D2)': 13,
    '中间支(RI)': 14,
    '第一钝缘支(OM1)': 15,
    '第二钝缘支(OM2)': 16,
    '左室侧后降支(L-PDA)': 17,
    '右室侧后降支(R-PDA)': 18,
    '左室侧后支(L-PLB)': 19,
    '右室侧后支(R-PLB)': 20,
    '右心室收缩压(RVSP)':21,
    '肺动脉收缩压(PASP)':22,
    '三尖瓣环收缩期位移(TAPSE)':23,
    '右心室游离壁基底段收缩期峰值速度(RV TDI s\')':24,
    '右心室射血分数(RV EF)':25,
    '右心室面积变化分数(RV FAC)':26,
    '右心室流出道血速度度积分(RVOT VTI)':27,
    '右心室每博量(RV SV)':28,
    '右心室每博量指数(RV SVi)':29,
    '右心室流出道峰值速度(RVOT peak vel)':30,
    '右心室流出道峰值压差(RVOT peak PG)':31,
    '右心室流出道平均压差(RVOT mean PG)':32,
    '右心室流出道加速时间(RVOT AccT)':33,
    '右心室游离壁纵向应变(RVFWS)':34,
    '右心室整体纵向应变(RVGLS)':35,
    '二尖瓣反流峰值速度(MR peak Vel)':36,
    '二尖瓣反流血速度度积分(MR VTI)':37,
    '二尖瓣反流峰值压差(MR peak PG)':38,
    '二尖瓣反流dp/dt(MR dp/dt)':39,
    '二尖瓣反流血流汇聚区直径(MR VC)':40,
    '二尖瓣反流血流汇聚区面积(3D MR VCA)':41,
    '二尖瓣反流PISA半径(MR PISA r)':42,
    '二尖瓣反流PISA混叠速度(MR PISA aliasing vel)':43,
    '二尖瓣反流PISA有效瓣口面积(MR PISA EROA)':44,
    '二尖瓣反流PISA有效瓣口面积(3D MR EROA)':45,
    '三尖瓣E峰速度(TV E)':46,
    '三尖瓣A峰速度(TV A)':47,
    '三尖瓣E/A比值(TV E/A)':48,
    '三尖瓣环脉冲多普勒速度(NA)':49,
    '三尖瓣环侧壁e\'速度(TV e\')':50,
    '三尖瓣E/e\'比值(TV E/e\')':51,
    '主肺动脉内径(mPA)':52,
    '右肺动脉内径(rPA)':53,
    '左肺动脉内径(lPA)':54
}

def create_formatted_df():
    """
    创建格式化的DataFrame
    前21行为固定的CTA冠脉节段数据，后续行从标准测量表.xlsx动态加载
    
    Returns:
        pd.DataFrame: 格式化后的数据框
    """
    # Define the column names
    columns = ["名称", "英文", "斑块种类", "类型", "症状", "数值", "单位", "狭窄程度", "闭塞"]

    # 前21行：固定的CTA冠脉节段数据
    fixed_cta_rows = [
        ["左主干近段(pLM)", "NA", "", "", "", "", "NA", "", "否"],
        ["左主干中段(mLM)", "NA", "", "", "", "", "NA", "", "否"],
        ["左主干远段(dLM)", "NA", "", "", "", "", "NA", "", "否"],
        ["左前降支近段(pLAD)", "NA", "", "", "", "", "NA", "", "否"],
        ["左前降支中段(mLAD)", "NA", "", "", "", "", "NA", "", "否"],
        ["左前降支远段(dLAD)", "NA", "", "", "", "", "NA", "", "否"],
        ["左回旋支近段(pLCX)", "NA", "", "", "", "", "NA", "", "否"],
        ["左回旋支中段(mLCX)", "NA", "", "", "", "", "NA", "", "否"],
        ["左回旋支远段(dLCX)", "NA", "", "", "", "", "NA", "", "否"],
        ["右冠近段(pRCA)", "NA", "", "", "", "", "NA", "", "否"],
        ["右冠中段(mRCA)", "NA", "", "", "", "", "NA", "", "否"],
        ["右冠远段(dRCA)", "NA", "", "", "", "", "NA", "", "否"],
        ["第一对角支(D1)", "NA", "", "", "", "", "NA", "", "否"],
        ["第二对角支(D2)", "NA", "", "", "", "", "NA", "", "否"],
        ["中间支(RI)", "NA", "", "", "", "", "NA", "", "否"],
        ["第一钝缘支(OM1)", "NA", "", "", "", "", "NA", "", "否"],
        ["第二钝缘支(OM2)", "NA", "", "", "", "", "NA", "", "否"],
        ["左室侧后降支(L-PDA)", "NA", "", "", "", "", "NA", "", "否"],
        ["右室侧后降支(R-PDA)", "NA", "", "", "", "", "NA", "", "否"],
        ["左室侧后支(L-PLB)", "NA", "", "", "", "", "NA", "", "否"],
        ["右室侧后支(R-PLB)", "NA", "", "", "", "", "NA", "", "否"]
    ]

    # 注释掉的原始后34行超声数据
    # ["右心室收缩压(RVSP)", "Right ventricular systolic pressure", "", "", "", "", "mmHg", "", "否"],
    # ["肺动脉收缩压(PASP)", "Pulmonary artery systolic pressure", "", "", "", "", "mmHg", "", "否"],
    # ... (其他33行)

    # 动态读取标准测量表.xlsx
    try:
        # 读取标准测量表.xlsx
        standard_df = pd.read_excel('data/标准测量表.xlsx')
        
        # 构建动态行数据
        dynamic_rows = []
        for _, row in standard_df.iterrows():
            # 跳过无效行
            if pd.isna(row['中文名称']) or row['中文名称'] == 'left':
                continue
                
            # 构建名称：中文名称(测量值简写)
            name = f"{row['中文名称']}({row['测量值简写'].strip()})"
            english = row['测量值名称'].strip()
            unit = row['单位'].strip() if pd.notna(row['单位']) else ""
            
            # 构建行数据，确保名称不出现在其他列中
            row_data = [
                name,           # 名称 (第1列)
                english,        # 英文 (第2列)
                "",            # 斑块种类 (第3列) - 空白
                "",            # 类型 (第4列) - 空白
                "",            # 症状 (第5列) - 空白
                "",            # 数值 (第6列) - 空白
                unit,          # 单位 (第7列)
                "",            # 狭窄程度 (第8列) - 空白
                "否"           # 闭塞 (第9列)
            ]
            dynamic_rows.append(row_data)
        
        print(f"✅ 从标准测量表.xlsx成功读取 {len(dynamic_rows)} 行数据")
        
    except Exception as e:
        print(f"⚠️ 读取标准测量表.xlsx失败: {e}")
        print("   使用空的动态数据")
        dynamic_rows = []

    # 合并固定行和动态行
    all_rows = fixed_cta_rows + dynamic_rows

    # 构建DataFrame
    df = pd.DataFrame(all_rows, columns=columns)
    
    # 打印表格到终端
    # print("\n📋 完整的表格结构：")
    # print("=" * 120)
    # print(f"总行数: {len(df)}")
    # print(f"- 前21行: 固定CTA冠脉节段")
    # print(f"- 后{len(dynamic_rows)}行: 从标准测量表.xlsx动态加载的超声数据")
    # print("\n表格内容:")
    # for i, row in df.iterrows():
    #     print(f"{i+1:2d}. {list(row)}")
    
    return df

def get_dynamic_row_index():
    """
    获取动态的行索引映射
    
    Returns:
        Dict[str, int]: 行索引映射字典
    """
    # 重新生成ROW_INDEX以匹配当前的DataFrame
    df = create_formatted_df()
    row_index = {}
    for idx, row in df.iterrows():
        row_index[row['名称']] = idx
    return row_index
