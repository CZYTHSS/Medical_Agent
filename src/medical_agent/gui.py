import tkinter as tk
from tkinter import ttk
import pandas as pd

# ---------------------------------------
# 1) Fixed coronary segments (table rows)
# ---------------------------------------
segments = [
    "左主干近段 (pLM)", "左主干中段 (mLM)", "左主干远段 (dLM)", 
    "左前降支近段 (pLAD)", "左前降支中段 (mLAD)", "左前降支远段 (dLAD)", 
    "左回旋支近段 (pLCX)", "左回旋支中段 (mLCX)", "左回旋支远段 (dLCX)",
    "右冠近段 (pRCA)", "右冠中段 (mRCA)", "右冠远段 (dRCA)",
    "第一对角支 (D1)", "第二对角支 (D2)", "中间支 (RI)",
    "第一钝缘支 (OM1)", "第二钝缘支 (OM2)", 
    "左室侧后降支 (L-PDA)", "右室侧降支 (L-PDA)",
    "左室侧后支 (L-PLB)", "右室侧后支 (R-PLB)", "右心室收缩压(RVSP)", "肺动脉收缩压(PASP)",
    "三尖瓣环收缩期位移(TAPSE)", "右心室游离壁基底段收缩期峰值速度(RV TDI s')", "右心室射血分数(RV EF)",
    "右心室面积变化分数(RV FAC)", "右心室流出道血速度度积分(RVOT VTI)", "右心室每博量(RV SV)",
    "右心室每博量指数(RV SVi)", "右心室流出道峰值速度(RVOT peak vel)", "右心室流出道峰值压差(RVOT peak PG)",
    "右心室流出道平均压差(RVOT mean PG)", "右心室流出道加速时间(RVOT AccT)", "右心室游离壁纵向应变(RVFWS)",
    "右心室整体纵向应变(RVGLS)", "二尖瓣反流峰值速度(MR peak Vel)", "二尖瓣反流血速度度积分(MR VTI)",
    "二尖瓣反流峰值压差(MR peak PG)", "二尖瓣反流dp/dt(MR dp/dt)", "二尖瓣反流血流汇聚区直径(MR VC)",
    "二尖瓣反流血流汇聚区面积(3D MR VCA)", "二尖瓣反流PISA半径(MR PISA r)", "二尖瓣反流PISA混叠速度(MR PISA aliasing vel)",
    "二尖瓣反流PISA有效瓣口面积(MR PISA EROA)", "二尖瓣反流PISA有效瓣口面积(3D MR EROA)", 
    "三尖瓣E峰速度(TV E)", "三尖瓣A峰速度(TV A)", "三尖瓣E/A比值(TV E/A)", "三尖瓣环脉冲多普勒速度(NA)",
    "三尖瓣环侧壁e'速度(TV e')", "三尖瓣E/e'比值(TV E/e')", "主肺动脉内径(mPA)", 
    "右肺动脉内径(rPA)", "左肺动脉内径(lPA)"
]

# -----------------------------------------------------
# 2) Table columns (after "冠脉节段") and dropdown setup
# -----------------------------------------------------
all_columns = ["名称", "英文", "斑块种类", "类型", "症状", "数值", "单位", "狭窄程度", "闭塞"]

dropdown_columns = ["斑块种类", "狭窄程度", "闭塞"]  # use Combobox
blank_columns = ["类型", "症状", "数值"]        # use Entry

dropdown_options = {
    "斑块种类": ["NONE", "软斑块（非钙化性斑块）", "混合密度斑块", "硬斑块（钙化性斑块）"],
    "狭窄程度": ["NONE", "局限性狭窄", "阶段性狭窄", "弥漫性狭窄"],
    "闭塞": ["NONE", "是", "否"]
}

# ----------------------------------------------------------------
# 3) Top boxes: "冠状动脉钙化总积分", "LM", "LAD", "LCX", "RCA", etc.
#    and dropdown for "冠状动脉起源、走形及终止" + "冠脉优势型"
# ----------------------------------------------------------------
def show_popup_with_df(df: pd.DataFrame, top_data: dict):
    """
    Creates a popup window with:
      - Top fields (冠状动脉钙化总积分, LM, LAD, LCX, RCA as Entry;
        冠状动脉起源、走形及终止, 冠脉优势型 as Combobox, 异常描述 as Entry)
      - Main table of segments vs. columns (some are dropdown, some are blank Entry)
    
    Args:
      df: DataFrame with columns [斑块种类, 类型, 症状, 大小(mm), 狭窄程度, 闭塞]
          and one row per segment.
      top_data: dict with keys:
        "冠状动脉钙化总积分", "LM", "LAD", "LCX", "RCA" (text)
        "冠状动脉起源、走形及终止" (dropdown: ["正常", "异常"])
        "冠脉优势型" (dropdown: ["右冠优势型", "左冠优势型", "均衡性"])
        "异常描述" (text)
    """
    root = tk.Tk()
    root.title("冠脉结构化填写")
    
    # Configure the style for Combobox widgets to improve text visibility
    style = ttk.Style()
    style.map('TCombobox', fieldbackground=[('readonly', 'white')], 
              selectbackground=[('readonly', '#0078d7')], 
              selectforeground=[('readonly', 'white')])
    style.configure('TCombobox', background='white', foreground='black')

    # =========================
    # Top Frame (boxes/fields)
    # =========================
    top_frame = ttk.Frame(root)
    top_frame.pack(padx=10, pady=5, fill="x")

    # Row 0: 冠状动脉钙化总积分, LM, LAD, LCX, RCA (all text fields)
    ttk.Label(top_frame, text="冠状动脉钙化总积分:").grid(row=0, column=0, sticky="e", padx=5)
    entry_calc = ttk.Entry(top_frame, width=10)
    entry_calc.grid(row=0, column=1)
    entry_calc.insert(0, top_data.get("冠状动脉钙化总积分", "NONE"))

    ttk.Label(top_frame, text="LM:").grid(row=0, column=2, sticky="e", padx=5)
    entry_lm = ttk.Entry(top_frame, width=10)
    entry_lm.grid(row=0, column=3)
    entry_lm.insert(0, top_data.get("LM", "NONE"))

    ttk.Label(top_frame, text="LAD:").grid(row=0, column=4, sticky="e", padx=5)
    entry_lad = ttk.Entry(top_frame, width=10)
    entry_lad.grid(row=0, column=5)
    entry_lad.insert(0, top_data.get("LAD", "NONE"))

    ttk.Label(top_frame, text="LCX:").grid(row=0, column=6, sticky="e", padx=5)
    entry_lcx = ttk.Entry(top_frame, width=10)
    entry_lcx.grid(row=0, column=7)
    entry_lcx.insert(0, top_data.get("LCX", "NONE"))

    ttk.Label(top_frame, text="RCA:").grid(row=0, column=8, sticky="e", padx=5)
    entry_rca = ttk.Entry(top_frame, width=10)
    entry_rca.grid(row=0, column=9)
    entry_rca.insert(0, top_data.get("RCA", "NONE"))

    # Row 1: 冠状动脉起源、走形及终止 (dropdown), 冠脉优势型 (dropdown)
    ttk.Label(top_frame, text="冠状动脉起源、走形及终止:").grid(row=1, column=0, sticky="e", padx=5)
    combo_origin = ttk.Combobox(top_frame, values=["NONE", "正常", "异常"], width=8, state="readonly")
    combo_origin.grid(row=1, column=1)
    combo_origin.set(top_data.get("冠状动脉起源、走形及终止", "NONE"))

    ttk.Label(top_frame, text="冠脉优势型:").grid(row=1, column=2, sticky="e", padx=5)
    combo_dominance = ttk.Combobox(top_frame, values=["NONE", "右冠优势型", "左冠优势型", "均衡性"], width=10, state="readonly")
    combo_dominance.grid(row=1, column=3)
    combo_dominance.set(top_data.get("冠脉优势型", "NONE"))

    # Row 2: 异常描述 (text field, spans entire row)
    ttk.Label(top_frame, text="异常描述:").grid(row=2, column=0, sticky="e", padx=5)
    entry_abnormal = ttk.Entry(top_frame, width=60)  # 宽度设置为60以占据整行
    entry_abnormal.grid(row=2, column=1, columnspan=9, sticky="w", padx=5)
    entry_abnormal.insert(0, top_data.get("异常描述", "NONE"))

    # =========================
    # Main Table (segments) with Scrollbars
    # =========================
    # Create a main frame for the table
    main_table_frame = ttk.Frame(root)
    main_table_frame.pack(padx=10, pady=5, fill="both", expand=True)

    # Create a canvas with scrollbars
    canvas = tk.Canvas(main_table_frame, width=800, height=400)
    v_scrollbar = ttk.Scrollbar(main_table_frame, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(main_table_frame, orient="horizontal", command=canvas.xview)
    
    # Create a frame inside the canvas for the actual table
    table_frame = ttk.Frame(canvas)
    
    # Configure scrolling
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    canvas.create_window((0, 0), window=table_frame, anchor="nw")
    
    # Pack scrollbars and canvas
    v_scrollbar.pack(side="right", fill="y")
    h_scrollbar.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    # Header row - 包含"名称"和其他8列，共9列
    all_df_columns = ["名称", "英文", "斑块种类", "类型", "症状", "数值", "单位", "狭窄程度", "闭塞"]
    for j, col in enumerate(all_df_columns):
        ttk.Label(table_frame, text=col, borderwidth=1, relief="solid", width=15)\
            .grid(row=0, column=j, sticky="nsew")

    # Data rows (55 rows total) - 直接使用DataFrame的所有列
    for i in range(len(df)):
        for j, col in enumerate(all_df_columns):
            raw_val = df.at[i, col] if (col in df.columns and i in df.index) else None
            value = str(raw_val) if pd.notna(raw_val) else "NONE"

            if col in dropdown_columns:
                # Combobox with improved visibility
                widget = ttk.Combobox(table_frame, width=13, state="readonly")
                widget['values'] = dropdown_options[col]
                widget.grid(row=i+1, column=j, sticky="nsew")
                widget.set(value)
            else:
                # Blank text field
                widget = ttk.Entry(table_frame, width=15)
                widget.grid(row=i+1, column=j, sticky="nsew")
                widget.insert(0, value)
    
    # Update canvas scroll region after adding all widgets
    def configure_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    table_frame.bind("<Configure>", configure_scroll_region)
    
    # Bind mouse wheel to canvas for scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind("<MouseWheel>", on_mousewheel)  # Windows
    canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
    canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux

    root.mainloop()

# -----------------------------------------------------
# Test with dummy data (top fields + main table "NONE")
# -----------------------------------------------------
def test_gui_with_dummy_df():
    # 1) Use the real formatted DataFrame instead of dummy data
    from table_format import create_formatted_df
    df = create_formatted_df()

    # 2) Build a dict for the top fields
    top_data = {
        "冠状动脉钙化总积分": "NONE",
        "LM": "NONE",
        "LAD": "NONE",
        "LCX": "NONE",
        "RCA": "NONE",
        "冠状动脉起源、走形及终止": "NONE",
        "冠脉优势型": "NONE",
        "异常描述": "NONE"
    }

    # 3) Show the popup
    show_popup_with_df(df, top_data)

if __name__ == "__main__":
    test_gui_with_dummy_df()
