import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy import stats
import datetime
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 数据库名称
DB_NAME = "probability_demo.db"

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    
    # 创建用户操作日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action_name TEXT NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME,
            duration INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

# 密码哈希处理
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 注册新用户
def register_user(username, password):
    if not username or not password:
        return False
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    hashed_pw = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# 用户认证
def authenticate(username, password):
    if not username or not password:
        return False
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    hashed_pw = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_pw))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# 记录操作开始
def record_operation_start(username, action_name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO user_operation_logs (username, action_name, start_time) VALUES (?, ?, ?)", 
                  (username, action_name, start_time))
    conn.commit()
    operation_id = cursor.lastrowid
    conn.close()
    return operation_id

# 记录操作结束
def record_operation_end(operation_id):
    if not operation_id:
        return
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("UPDATE user_operation_logs SET end_time = ?, duration = strftime('%s', ?) - strftime('%s', start_time) WHERE id = ?", 
                  (end_time, end_time, operation_id))
    conn.commit()
    conn.close()

# 清除用户历史操作记录
def clear_user_history(username, current_login_time):
    if not username or not current_login_time:
        return False
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        # 清除在当前登录时间之前完成的操作记录
        cursor.execute("""
            DELETE FROM user_operation_logs 
            WHERE username = ? AND end_time IS NOT NULL AND end_time < ?
        """, (username, current_login_time))
        conn.commit()
        return True
    except:
        conn.rollback()
        return False
    finally:
        conn.close()
        
# 获取用户操作记录
def get_user_actions(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT action_name, start_time, end_time, duration 
        FROM user_operation_logs 
        WHERE username = ? AND end_time IS NOT NULL
    ''', (username,))
    actions = cursor.fetchall()
    conn.close()
    return actions

# 创建导出Excel文件的函数
def export_operation_logs_to_excel(username, current_login_time):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 获取当前登录时间之后的操作记录
    cursor.execute('''
        SELECT action_name, start_time, end_time, duration 
        FROM user_operation_logs 
        WHERE username = ? AND start_time >= ?
    ''', (username, current_login_time))
    user_actions = cursor.fetchall()
    conn.close()
    
    # 创建DataFrame
    df = pd.DataFrame(user_actions, columns=["操作名称", "开始时间", "结束时间", "持续时间"])
    
    # 创建Excel文件
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='操作记录')
    
    return output.getvalue()

# 主界面
def main_page():
    st.title("欢迎来到概率统计动态演示平台")
    st.markdown("""
    **平台功能**：
    - 交互式概率实验模拟
    - 统计分布可视化
    - 假设检验与参数估计演示
    - 实验数据记录与分析
    """)
    
    st.subheader("章节概览")
    chapters = {
        "第一章": "随机事件及其概率",
        "第二章": "随机变量及其分布",
        "第三章": "多维随机变量及其分布",
        "第四章": "随机变量的数字特征",
        "第五章": "中心极限定理演示",
        "第六章": "统计量与抽样分布",
        "第七章": "点估计",
        "第八章": "区间估计",
        "第九章": "假设检验"
    }
    
    # 优化章节概览的显示方式，使用两列布局
    cols = st.columns(2)
    for i, (chap, desc) in enumerate(chapters.items()):
        with cols[i % 2]:  # 交替放入两列
            st.markdown(f"**{chap}**：{desc}")

# 个人账号信息函数
def user_center():
    st.subheader("个人账号信息")
    if st.session_state.logged_in:
        if st.session_state.username != "游客":
            # 确保 current_login_time 已初始化
            if 'current_login_time' not in st.session_state:
                st.session_state.current_login_time = st.session_state.login_time
            
            # 显示用户信息
            user_info = f"""
                <style>
                    .personal-info {{
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto;
                        max-width: 800px;
                    }}
                    .personal-info ul {{
                        list-style-type: none;
                        padding: 0;
                    }}
                    .personal-info li {{
                        margin: 10px 0;
                    }}
                </style>
                <div class="personal-info">
                    <ul>
                        <li><strong>用户名</strong>: {st.session_state.username}</li>
                        <li><strong>登录时间</strong>: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    </ul>
                </div>
            """
            st.markdown(user_info, unsafe_allow_html=True)
            
            # 添加清除历史记录按钮
            if st.button("清除历史记录"):
                if clear_user_history(st.session_state.username, st.session_state.current_login_time):
                    st.success("历史记录已清除！")
                    st.session_state.show_history = False
                    st.rerun()
                else:
                    st.error("清除历史记录失败！")
            
            # 添加历史操作记录按钮
            if 'show_history' not in st.session_state:
                st.session_state.show_history = False
                
            if st.button("显示历史操作记录"):
                st.session_state.show_history = not st.session_state.show_history
                
            # 添加导出操作记录为Excel按钮
            if st.button("导出操作记录"):
                excel_data = export_operation_logs_to_excel(st.session_state.username, st.session_state.current_login_time)
                st.download_button(
                    label="下载Excel文件",
                    data=excel_data,
                    file_name="操作记录.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            # 显示操作记录
            if st.session_state.show_history:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT action_name, start_time, end_time, duration 
                    FROM user_operation_logs 
                    WHERE username = ? AND end_time < ?
                ''', (st.session_state.username, st.session_state.current_login_time))
                user_actions = cursor.fetchall()
                conn.close()
                
                if user_actions:
                    # 将历史操作记录转换为DataFrame并显示为表格
                    df = pd.DataFrame(user_actions, columns=["操作名称", "开始时间", "结束时间", "持续时间(秒)"])
                    st.dataframe(df)
                else:
                    st.info("暂无历史操作记录")
            else:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT action_name, start_time, end_time, duration 
                    FROM user_operation_logs 
                    WHERE username = ? AND start_time >= ?
                ''', (st.session_state.username, st.session_state.current_login_time))
                user_actions = cursor.fetchall()
                conn.close()
                
                if user_actions:
                    # 将本次操作记录转换为DataFrame并显示为表格
                    df = pd.DataFrame(user_actions, columns=["操作名称", "开始时间", "结束时间", "持续时间(秒)"])
                    st.dataframe(df)
                else:
                    st.info("暂无本次操作记录")
        else:
            st.write("游客登录")
    else:
        st.write("未登录")
        
    
# 第一章内容：随机事件及其概率
def chapter1():
    st.header("第一章 随机事件及其概率")
    
    # 投币实验
    with st.expander("投币实验", expanded=True):
        st.subheader("实验目的")
        st.write("通过模拟投币实验，验证概率的统计定义：当试验次数足够多时，事件发生的频率会稳定在其概率附近。")
        
        st.subheader("实验内容")
        st.write("模拟投掷均匀硬币的过程，记录正面和反面出现的次数，计算其频率并观察随着实验次数增加频率的变化趋势。")
        
        n_coin = st.slider("投币次数", 1, 1000, 100)
        if st.button("执行投币实验", key="coin_toss"):
            operation_id = record_operation_start(st.session_state.username, "执行投币实验")
            results = ["正面" if np.random.random() > 0.5 else "反面" for _ in range(n_coin)]
            heads = results.count("正面")
            tails = results.count("反面")
            
            st.subheader("实验结果")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("正面次数", heads)
                st.metric("正面频率", f"{heads/n_coin:.2%}")
            with col2:
                st.metric("反面次数", tails)
                st.metric("反面频率", f"{tails/n_coin:.2%}")
            
            fig, ax = plt.subplots()
            ax.bar(["正面", "反面"], [heads, tails], color=['red', 'blue'])
            ax.set_title("投币结果分布")
            ax.set_ylabel("出现次数")
            st.pyplot(fig)
            
            st.subheader("实验分析")
            freq_heads = heads / n_coin
            freq_tails = tails / n_coin
            st.write(f"在{n_coin}次投币实验中，正面出现频率为{freq_heads:.2%}，反面出现频率为{freq_tails:.2%}。")
            st.write("随着实验次数增加，正面和反面出现的频率会逐渐接近理论概率50%，这验证了大数定律。")
            
            record_operation_end(operation_id)

    # 布丰投针实验
    with st.expander("布丰投针实验"):
        st.subheader("实验目的")
        st.write("通过布丰投针实验，理解几何概率的概念，并利用实验结果估算圆周率π的值。")
        
        st.subheader("实验内容")
        st.write("模拟向一组等距平行线投掷细针的过程，记录针与直线相交的次数，利用几何概率原理计算π的估计值。")
        
        st.markdown("""**理论公式**：P = (2L)/(πD) → π ≈ (2L×n)/(D×c)  
        其中：L为针长，D为平行线间距，n为投针次数，c为相交次数""")
        
        n_needle = st.slider("投针次数", 10, 5000, 1000)
        L = st.slider("针长度", 0.1, 2.0, 1.0)
        D = st.slider("平行线间距", 0.5, 3.0, 2.0)
        
        if st.button("执行投针实验", key="needle_toss"):
            operation_id = record_operation_start(st.session_state.username, "执行布丰投针实验")
            crosses = 0
            for _ in range(n_needle):
                y = np.random.uniform(0, D/2)
                theta = np.random.uniform(0, np.pi/2)
                if y <= (L/2)*np.sin(theta):
                    crosses += 1
            
            st.subheader("实验结果")
            if crosses > 0:
                pi_estimate = (2*L*n_needle)/(D*crosses)
                
                st.metric("π估计值", f"{pi_estimate:.6f}", 
                         delta=f"误差:{abs(pi_estimate-np.pi)/np.pi:.2%}")
                
                fig, ax = plt.subplots()
                x = np.linspace(0, n_needle, n_needle)
                y_est = [(2*L*i)/(D*max(1,c)) for i, c in enumerate(range(1, n_needle+1),1)]
                ax.plot(x, y_est, label="估计值")
                ax.axhline(y=np.pi, color='r', linestyle='--', label="真实值")
                ax.set_title("π值估计收敛过程")
                ax.set_xlabel("投针次数")
                ax.set_ylabel("π估计值")
                ax.legend()
                st.pyplot(fig)
                
                st.subheader("实验分析")
                st.write(f"在{n_needle}次投针实验中，针与直线相交{crosses}次，得到π的估计值为{float(pi_estimate):.6f}。")
                st.write("随着投针次数增加，π的估计值会逐渐接近真实值，这展示了几何概率在实际问题中的应用。")
            else:
                st.warning("投针实验中没有发生相交情况，无法估计π值。请增加投针次数或调整参数后重试。")
            
            record_operation_end(operation_id)

    # 事件间的关系
    with st.expander("事件间的关系"):
        st.subheader("实验目的")
        st.write("通过可视化方式展示事件之间的关系（如并集、差集等），理解概率的基本运算法则。")
        
        st.subheader("实验内容")
        st.write("设定两个事件A和B的概率，计算并可视化展示它们的并集、差集等关系，验证概率加法公式。")
        
        col1, col2 = st.columns(2)
        with col1:
            probA = st.slider("事件A的概率", 0.0, 1.0, 0.5)
        with col2:
            probB = st.slider("事件B的概率", 0.0, 1.0, 0.5)
        
        # 计算最大可能的交集概率
        max_intersection = min(probA, probB)
        probAB = st.slider("事件A∩B的概率", 0.0, max_intersection, min(0.3, max_intersection))
        
        if st.button("显示事件关系", key="event_relation"):
            operation_id = record_operation_start(st.session_state.username, "显示事件关系")
            
            # 计算概率
            union_prob = probA + probB - probAB  # 并集概率
            diff_prob = max(0, probA - probB)    # 差集概率
            cond_prob = probAB / probA if probA > 0 else 0  # 条件概率
            
            st.subheader("实验结果")
            # 绘制维恩图示意
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 并集维恩图
            ax1.set_title(f"事件A∪B的概率: {union_prob:.2f}")
            circle1 = plt.Circle((0.3, 0.5), 0.3, alpha=0.5, color='blue')
            circle2 = plt.Circle((0.7, 0.5), 0.3, alpha=0.5, color='red')
            ax1.add_patch(circle1)
            ax1.add_patch(circle2)
            ax1.text(0.1, 0.5, "A", fontsize=12)
            ax1.text(0.9, 0.5, "B", fontsize=12)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            
            # 差集维恩图
            ax2.set_title(f"事件A-B的概率: {diff_prob:.2f}")
            circle1 = plt.Circle((0.3, 0.5), 0.3, alpha=0.5, color='blue')
            circle2 = plt.Circle((0.7, 0.5), 0.3, alpha=0.5, color='white', edgecolor='red')
            ax2.add_patch(circle1)
            ax2.add_patch(circle2)
            ax2.text(0.1, 0.5, "A", fontsize=12)
            ax2.text(0.9, 0.5, "B", fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            st.pyplot(fig)
            
            # 显示概率计算结果
            st.subheader("概率计算结果")
            result_df = pd.DataFrame({
                "概率类型": ["P(A)", "P(B)", "P(A∩B)", "P(A∪B)", "P(A-B)", "P(B|A)"],
                "数值": [
                    f"{probA:.4f}",
                    f"{probB:.4f}",
                    f"{probAB:.4f}",
                    f"{union_prob:.4f}",
                    f"{diff_prob:.4f}",
                    f"{cond_prob:.4f}"
                ]
            })
            st.dataframe(result_df)
            
            st.subheader("实验分析")
            st.write(f"事件A的概率为{ probA }，事件B的概率为{ probB }。")
            st.write(f"A和B的并集概率为{ union_prob }，符合概率加法公式：P(A∪B) = P(A) + P(B) - P(A∩B)")
            st.write(f"A与B的差集概率为{ diff_prob }，表示事件A发生而B不发生的概率。")
            
            record_operation_end(operation_id)

    # 骰子实验
    with st.expander("骰子实验"):
        st.subheader("实验目的")
        st.write("通过模拟掷骰子实验，理解离散型均匀分布的特点，观察随机现象的统计规律性。")
        
        st.subheader("实验内容")
        st.write("模拟投掷均匀六面骰子的过程，记录每个点数出现的次数和频率，验证各点数出现的概率是否相等。")
        
        n_dice = st.slider("投掷次数", 10, 1000, 100)
        
        if st.button("执行骰子实验", key="dice_roll"):
            operation_id = record_operation_start(st.session_state.username, "执行骰子实验")
            results = [np.random.randint(1, 7) for _ in range(n_dice)]
            counts = [results.count(i) for i in range(1, 7)]
            
            st.subheader("实验结果")
            fig, ax = plt.subplots()
            ax.bar(range(1, 7), counts, color='skyblue')
            ax.set_title("骰子投掷结果分布")
            ax.set_xlabel("骰子点数")
            ax.set_ylabel("出现次数")
            st.pyplot(fig)
            
            # 显示结果表格
            st.subheader("投掷结果统计")
            df = pd.DataFrame({
                "点数": range(1, 7),
                "频数": counts,
                "频率": [f"{c/n_dice:.2%}" for c in counts]
            })
            st.dataframe(df)
            
            st.subheader("实验分析")
            st.write(f"在{ n_dice }次掷骰子实验中，各个点数出现的频率大致相等，接近理论概率1/6（约16.67%）。")
            st.write("随着实验次数增加，各点数出现的频率会更加接近，体现了均匀分布的特性。")
            
            record_operation_end(operation_id)

    # 几何概型
    with st.expander("几何概型"):
        st.subheader("实验目的")
        st.write("通过面积模型理解几何概型的基本概念，即事件发生的概率与构成该事件的区域长度(面积或体积)成正比。")
        
        st.subheader("实验内容")
        st.write("在单位正方形内定义一个矩形区域，计算该矩形面积与正方形面积的比值，以此理解几何概率的计算方法。")
        
        a = st.slider("矩形宽度", 0.1, 1.0, 0.5)
        b = st.slider("矩形高度", 0.1, 1.0, 0.5)
        
        if st.button("显示几何概型", key="geo_prob"):
            operation_id = record_operation_start(st.session_state.username, "显示几何概型")
            prob = a * b  # 面积概率
            
            st.subheader("实验结果")
            fig, ax = plt.subplots()
            # 绘制大正方形
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black'))
            # 绘制概率区域
            ax.add_patch(Rectangle((0, 0), a, b, fill=True, color='blue', alpha=0.5))
            ax.text(a/2, b/2, f"P = {prob:.2f}", ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title("几何概型面积演示")
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"在单位正方形中，宽度为{ a }、高度为{ b }的矩形区域面积为{ prob }。")
            st.write("在几何概型中，随机点落入该矩形区域的概率等于矩形面积与正方形面积之比，即等于矩形面积本身。")
            
            record_operation_end(operation_id)

    # 条件概率和独立性判断
    with st.expander("条件概率和独立性判断"):
        st.subheader("实验目的")
        st.write("理解条件概率的概念和计算方法，掌握判断两个事件是否独立的方法。")
        
        st.subheader("实验内容")
        st.write("给定两个事件A和B的概率及其联合概率，计算条件概率P(B|A)，并判断事件A和B是否独立。")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pA = st.slider("P(A)", 0.01, 0.99, 0.5)
        with col2:
            pB = st.slider("P(B)", 0.01, 0.99, 0.5)
        with col3:
            max_pAB = min(pA, pB)
            pAB = st.slider("P(A∩B)", 0.01, max_pAB, min(0.25, max_pAB))
        
        if st.button("计算条件概率", key="cond_prob"):
            operation_id = record_operation_start(st.session_state.username, "计算条件概率")
            # 计算条件概率
            pBA = pAB / pA  # P(B|A)
            pAB_indep = pA * pB  # 独立情况下的P(A∩B)
            is_independent = abs(pAB - pAB_indep) < 0.01
            
            st.subheader("实验结果")
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.metric("P(B|A)", f"{pBA:.4f}")
                st.metric("P(A∩B) (独立假设)", f"{pAB_indep:.4f}")
            with col2:
                st.metric("事件A和B是否独立", "是" if is_independent else "否")
                st.metric("P(A∪B)", f"{pA + pB - pAB:.4f}")
            
            # 绘制概率条形图
            fig, ax = plt.subplots()
            labels = ["P(A)", "P(B)", "P(A∩B)"]
            values = [pA, pB, pAB]
            ax.bar(labels, values, color=['blue', 'green', 'red'])
            ax.set_title("概率分布")
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"条件概率P(B|A) = {pBA:.4f}，表示在事件A发生的条件下，事件B发生的概率。")
            st.write(f"根据独立性定义，如果P(A∩B) ≈ P(A)×P(B)，则事件A和B独立。本实验中{'满足' if is_independent else '不满足'}这一条件。")
            st.write("独立事件意味着一个事件的发生不影响另一个事件发生的概率。")
            
            record_operation_end(operation_id)

# 第二章内容：随机变量及其分布
def chapter2():
    st.header("第二章 随机变量及其分布")
    
    # 两点分布
    with st.expander("两点分布", expanded=True):
        st.subheader("实验目的")
        st.write("理解两点分布（伯努利分布）的概念和特点，掌握其概率质量函数的形式和参数意义。")
        
        st.subheader("实验内容")
        st.write("两点分布是一种最简单的离散概率分布，随机变量只有两个可能的取值：0和1。本实验通过设置成功概率p，观察两点分布的概率分布情况。")
        
        p = st.slider("成功概率p", 0.01, 0.99, 0.5,key="bernoulli_p")
        
        if st.button("显示两点分布", key="bernoulli"):
            operation_id = record_operation_start(st.session_state.username, "显示两点分布")
            # 计算概率
            x = [0, 1]
            probs = [1-p, p]
            
            st.subheader("实验结果")
            # 绘制两点分布
            fig, ax = plt.subplots()
            ax.bar(x, probs, color='skyblue')
            ax.set_title(f"两点分布 PMF (p = {p:.2f})")
            ax.set_xlabel("X")
            ax.set_ylabel("概率")
            ax.set_xticks(x)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"两点分布中，随机变量X=1（成功）的概率为p={p:.2f}，X=0（失败）的概率为1-p={1-p:.2f}。")
            st.write("两点分布适用于描述只有两种可能结果的随机试验，如抛硬币、产品合格与否等场景。")
            
            record_operation_end(operation_id)

    # 二项分布
    with st.expander("二项分布"):
        st.subheader("实验目的")
        st.write("理解二项分布的概念和性质，掌握其累积分布函数的特点，了解二项分布与伯努利分布的关系。")
        
        st.subheader("实验内容")
        st.write("二项分布描述了n次独立重复伯努利试验中成功次数的概率分布。本实验通过设置试验次数n和成功概率p，观察二项分布的累积分布函数。")
        
        n = st.slider("试验次数n", 1, 50, 10,key="binom_n")
        p = st.slider("成功概率p", 0.01, 0.99, 0.5, key="binom_p")
        
        if st.button("显示二项分布", key="binomial"):
            operation_id = record_operation_start(st.session_state.username, "显示二项分布")
            # 计算二项分布CDF
            k = np.arange(0, n+1)
            cdf = np.cumsum(stats.binom.pmf(k, n, p))
            
            st.subheader("实验结果")
            # 绘制CDF
            fig, ax = plt.subplots()
            ax.step(k, cdf, where='post')
            ax.set_title(f"二项分布CDF (n = {n}, p = {p:.2f})")
            ax.set_xlabel("成功次数k")
            ax.set_ylabel("累积概率")
            ax.set_xticks(k)
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"二项分布B(n={n}, p={p:.2f})描述了n次独立试验中成功次数的概率分布。")
            st.write("累积分布函数(CDF)表示成功次数不超过k的概率。从图中可以看出，随着成功次数k的增加，累积概率逐渐增加到1。")
            st.write("二项分布的期望为np，方差为np(1-p)，当n较大时可近似为正态分布。")
            
            record_operation_end(operation_id)

    # 泊松分布
    with st.expander("泊松分布"):
        st.subheader("实验目的")
        st.write("理解泊松分布的概念和应用场景，掌握其概率质量函数的特点及参数λ的意义。")
        
        st.subheader("实验内容")
        st.write("泊松分布常用于描述单位时间或空间内随机事件发生次数的概率分布。本实验通过设置参数λ，观察泊松分布的概率质量函数。")
        
        lam = st.slider("λ值", 0.1, 20.0, 5.0)
        
        if st.button("显示泊松分布", key="poisson"):
            operation_id = record_operation_start(st.session_state.username, "显示泊松分布")
            # 计算泊松分布概率
            x = np.arange(0, 31)
            probs = stats.poisson.pmf(x, lam)
            
            st.subheader("实验结果")
            # 绘制泊松分布
            fig, ax = plt.subplots()
            ax.bar(x, probs, color='skyblue')
            ax.set_title(f"泊松分布 PMF (λ = {lam:.2f})")
            ax.set_xlabel("事件发生次数")
            ax.set_ylabel("概率")
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"泊松分布P(λ={lam:.2f})描述了单位时间内事件发生次数的概率分布。")
            st.write("泊松分布的期望和方差均为λ，分布呈现单峰形态，随着λ增大，分布逐渐接近正态分布。")
            st.write("泊松分布常用于描述稀有事件的发生次数，如交通事故数、电话呼叫次数等。")
            
            record_operation_end(operation_id)

    # 正态分布
    with st.expander("正态分布"):
        st.subheader("实验目的")
        st.write("理解正态分布的概念和性质，掌握其概率密度函数的特点，了解均值μ和标准差σ对分布形态的影响。")
        
        st.subheader("实验内容")
        st.write("正态分布是最常见的连续型概率分布，具有对称的钟形曲线。本实验通过调整均值μ和标准差σ，观察正态分布的概率密度函数变化。")
        
        mean = st.slider("均值μ", -5.0, 5.0, 0.0)
        std = st.slider("标准差σ", 0.1, 3.0, 1.0)
        
        if st.button("显示正态分布", key="normal"):
            operation_id = record_operation_start(st.session_state.username, "显示正态分布")
            # 生成正态分布数据
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = stats.norm.pdf(x, mean, std)
            
            st.subheader("实验结果")
            # 绘制正态分布曲线
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.fill_between(x, y, alpha=0.3)
            ax.set_title(f"正态分布 PDF (μ = {mean}, σ = {std})")
            ax.set_xlabel("X")
            ax.set_ylabel("概率密度")
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"正态分布N(μ={mean}, σ={std})呈现对称的钟形曲线，均值μ决定了曲线的中心位置，标准差σ决定了曲线的分散程度。")
            st.write("正态分布具有3σ原则：约99.7%的数据落在[μ-3σ, μ+3σ]区间内。")
            st.write("许多自然现象和社会现象都近似服从正态分布，是统计推断的重要基础。")
            
            record_operation_end(operation_id)

    # 均匀分布
    with st.expander("均匀分布"):
        st.subheader("实验目的")
        st.write("理解均匀分布的概念和特点，掌握其概率密度函数的形式，了解均匀分布在随机模拟中的应用。")
        
        st.subheader("实验内容")
        st.write("均匀分布是一种简单的连续型概率分布，在指定区间内各点的概率密度相等。本实验通过设置区间范围，观察均匀分布的概率密度函数。")
        
        min_val = st.slider("最小值", -5.0, 0.0, 0.0)
        max_val = st.slider("最大值", 0.1, 5.0, 1.0)
        
        if st.button("显示均匀分布", key="uniform"):
            operation_id = record_operation_start(st.session_state.username, "显示均匀分布")
            # 生成均匀分布数据
            x = np.linspace(min_val - 1, max_val + 1, 100)
            y = stats.uniform.pdf(x, loc=min_val, scale=max_val - min_val)
            
            st.subheader("实验结果")
            # 绘制均匀分布曲线
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.fill_between(x, y, alpha=0.3)
            ax.set_title(f"均匀分布 PDF (min = {min_val}, max = {max_val})")
            ax.set_xlabel("X")
            ax.set_ylabel("概率密度")
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"均匀分布U({min_val}, {max_val})在区间[{min_val}, {max_val}]内的概率密度为常数1/({max_val}-{min_val})，区间外的概率密度为0。")
            st.write(f"该分布的期望为({min_val}+{max_val})/2，方差为({max_val}-{min_val})²/12。")
            st.write("均匀分布常用于随机数生成、蒙特卡洛模拟等领域。")
            
            record_operation_end(operation_id)

    # 指数分布
    with st.expander("指数分布"):
        st.subheader("实验目的")
        st.write("理解指数分布的概念和应用场景，掌握其概率密度函数的特点，了解指数分布的无记忆性。")
        
        st.subheader("实验内容")
        st.write("指数分布常用于描述两个连续事件发生的时间间隔的概率分布。本实验通过设置率参数λ，观察指数分布的概率密度函数。")
        
        rate = st.slider("率参数λ", 0.1, 2.0, 1.0)
        
        if st.button("显示指数分布", key="exponential"):
            operation_id = record_operation_start(st.session_state.username, "显示指数分布")
            # 生成指数分布数据
            x = np.linspace(0, 5, 100)
            y = stats.expon.pdf(x, scale=1/rate)
            
            st.subheader("实验结果")
            # 绘制指数分布曲线
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.fill_between(x, y, alpha=0.3)
            ax.set_title(f"指数分布 PDF (λ = {rate:.2f})")
            ax.set_xlabel("X")
            ax.set_ylabel("概率密度")
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"指数分布Exp(λ={rate:.2f})的概率密度函数随x增大而指数衰减，描述了事件发生的时间间隔分布。")
            st.write(f"该分布的期望和标准差均为1/λ = {1/rate:.2f}，方差为1/λ²。")
            st.write("指数分布具有无记忆性，即P(X > s+t | X > s) = P(X > t)，适用于描述电子元件寿命、排队等待时间等。")
            
            record_operation_end(operation_id)

    # 泊松分布近似二项分布
    with st.expander("泊松分布近似二项分布"):
        st.subheader("实验目的")
        st.write("理解泊松定理，即当n很大且p很小时，二项分布可以用泊松分布近似，掌握这一近似的应用条件。")
        
        st.subheader("实验内容")
        st.write("当二项分布的试验次数n很大而成功概率p很小时，二项分布B(n,p)可以用泊松分布P(λ=np)近似。本实验通过对比两种分布，验证这一近似效果。")
        
        n = st.slider("试验次数n", 10, 1000, 100, key="poisson_approx_n")
        p = st.slider("成功概率p", 0.01, 0.1, 0.05, key="poisson_approx_p")
        
        if st.button("显示近似效果", key="poisson_approx"):
            operation_id = record_operation_start(st.session_state.username, "显示泊松近似二项分布")
            lam = n * p  # 泊松分布参数
            x = np.arange(0, min(n, int(lam*3) + 1))
            
            # 计算概率
            binom_probs = stats.binom.pmf(x, n, p)
            poisson_probs = stats.poisson.pmf(x, lam)
            
            st.subheader("实验结果")
            # 绘制比较图
            fig, ax = plt.subplots()
            ax.plot(x, binom_probs, 'b-', label="二项分布")
            ax.plot(x, poisson_probs, 'r--', label="泊松分布")
            ax.set_title(f"二项分布与泊松分布比较 (λ = {lam:.2f})")
            ax.set_xlabel("成功次数k")
            ax.set_ylabel("概率")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"当n={n}较大且p={p:.2f}较小时，二项分布B(n,p)与泊松分布P(λ={lam:.2f})的概率分布非常接近。")
            st.write("这一近似在实际应用中很有用，可以简化计算，特别是当n很大时，直接计算二项分布概率会比较复杂。")
            
            record_operation_end(operation_id)

    # 二项分布近似正态分布
    with st.expander("二项分布近似正态分布"):
        st.subheader("实验目的")
        st.write("理解棣莫弗-拉普拉斯中心极限定理，即当n足够大时，二项分布可以用正态分布近似，掌握这一近似的应用条件。")
        
        st.subheader("实验内容")
        st.write("当二项分布的试验次数n足够大，且np和n(1-p)都较大时，二项分布B(n,p)可以用正态分布N(np, np(1-p))近似。本实验通过对比两种分布，验证这一近似效果。")
        
        n = st.slider("试验次数n", 10, 1000, 50, key="normal_approx_n")
        p = st.slider("成功概率p", 0.1, 0.9, 0.5, key="normal_approx_p")
        
        if st.button("显示近似效果", key="normal_approx"):
            operation_id = record_operation_start(st.session_state.username, "显示正态近似二项分布")
            # 二项分布参数
            mean = n * p
            std = np.sqrt(n * p * (1 - p))
            x = np.arange(0, n+1)
            
            # 计算概率
            binom_probs = stats.binom.pmf(x, n, p)
            normal_probs = stats.norm.pdf(x, mean, std)
            
            st.subheader("实验结果")
            # 绘制比较图
            fig, ax = plt.subplots()
            ax.bar(x, binom_probs, alpha=0.6, label="二项分布")
            ax.plot(x, normal_probs, 'r-', label="正态分布")
            ax.set_title(f"二项分布与正态分布比较 (n = {n}, p = {p:.2f})")
            ax.set_xlabel("成功次数k")
            ax.set_ylabel("概率")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("实验分析")
            st.write(f"当n={n}足够大时，二项分布B(n={n}, p={p:.2f})可以用正态分布N(μ={mean:.2f}, σ={std:.2f})近似。")
            st.write("这一近似是中心极限定理的一个具体应用，使得我们可以利用正态分布的性质来近似计算二项分布的概率。")
            st.write("一般来说，当np ≥ 5且n(1-p) ≥ 5时，这个近似效果较好。")
            
            record_operation_end(operation_id)

# 第三章内容：多维随机变量及其分布
def chapter3():
    st.header("第三章 多维随机变量及其分布")
    
    # 二维离散型随机变量
    with st.expander("二维离散型随机变量", expanded=True):
        st.subheader("实验目的")
        st.write("理解二维离散型随机变量的联合分布、边缘分布的概念和性质，掌握它们之间的关系。")
        
        st.subheader("实验内容")
        st.write("对于二维离散型随机变量(X,Y)，本实验通过设置联合概率分布，计算并可视化展示X和Y的边缘分布，理解联合分布与边缘分布的关系。")
        
        # 获取用户输入的概率
        col1, col2 = st.columns(2)
        with col1:
            prob00 = st.slider("P(X=0,Y=0)", 0.01, 0.95, 0.25)
            prob01 = st.slider("P(X=0,Y=1)", 0.01, 0.95, 0.25)
        with col2:
            prob10 = st.slider("P(X=1,Y=0)", 0.01, 0.95, 0.25)
            prob11 = st.slider("P(X=1,Y=1)", 0.01, 0.95, 0.25)
        
        if st.button("更新联合分布", key="joint_dist_update"):
            operation_id = record_operation_start(st.session_state.username, "更新二维离散分布")
            # 标准化概率确保和为1
            total = prob00 + prob01 + prob10 + prob11
            p00 = prob00 / total
            p01 = prob01 / total
            p10 = prob10 / total
            p11 = prob11 / total
            
            # 创建联合分布数据
            joint_data = pd.DataFrame({
                "X": [0, 0, 1, 1],
                "Y": [0, 1, 0, 1],
                "概率": [p00, p01, p10, p11]
            })
            
            # 计算边缘分布
            marginal_x = joint_data.groupby("X")["概率"].sum().reset_index()
            marginal_y = joint_data.groupby("Y")["概率"].sum().reset_index()
            
            # 计算条件概率
            cond_prob_y1_x0 = p01 / (p00 + p01) if (p00 + p01) > 0 else 0
            cond_prob_y0_x1 = p10 / (p10 + p11) if (p10 + p11) > 0 else 0
            
            st.subheader("实验结果")
            # 显示联合分布热力图
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            heat_data = np.array([[p00, p01], [p10, p11]])
            im = ax1.imshow(heat_data, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax1.text(j, i, f"{heat_data[i, j]:.2f}", ha="center", va="center", color="black")
            ax1.set_xticks([0, 1])
            ax1.set_yticks([0, 1])
            ax1.set_xticklabels(["Y=0", "Y=1"])
            ax1.set_yticklabels(["X=0", "X=1"])
            ax1.set_title("联合分布 P(X,Y)")
            fig1.colorbar(im)
            st.pyplot(fig1)
            
            # 显示边缘分布
            col1, col2 = st.columns(2)
            with col1:
                fig2, ax2 = plt.subplots()
                ax2.bar(marginal_x["X"], marginal_x["概率"], color="skyblue")
                ax2.set_title("X的边缘分布")
                ax2.set_xticks([0, 1])
                ax2.set_ylim(0, 1)
                st.pyplot(fig2)
            
            with col2:
                fig3, ax3 = plt.subplots()
                ax3.bar(marginal_y["Y"], marginal_y["概率"], color="lightgreen")
                ax3.set_title("Y的边缘分布")
                ax3.set_xticks([0, 1])
                ax3.set_ylim(0, 1)
                st.pyplot(fig3)
            
            # 显示条件概率
            st.subheader("条件概率计算")
            cond_df = pd.DataFrame({
                "条件概率": ["P(Y=1|X=0)", "P(Y=0|X=1)"],
                "数值": [f"{cond_prob_y1_x0:.4f}", f"{cond_prob_y0_x1:.4f}"]
            })
            st.dataframe(cond_df)
            
            st.subheader("实验分析")
            st.write("联合分布描述了二维随机变量(X,Y)取每个可能值的概率，而边缘分布则是单个随机变量的概率分布。")
            st.write("X的边缘分布是通过对Y的所有可能值求和得到的，Y的边缘分布同理。")
            st.write("本实验中，X的边缘分布为P(X=0)=%.2f, P(X=1)=%.2f" % (marginal_x["概率"][0], marginal_x["概率"][1]))
            st.write("Y的边缘分布为P(Y=0)=%.2f, P(Y=1)=%.2f" % (marginal_y["概率"][0], marginal_y["概率"][1]))
            
            record_operation_end(operation_id)

    # 二维连续型随机变量
    with st.expander("二维连续型随机变量"):
        st.subheader("实验目的")
        st.write("理解二维连续型随机变量的联合分布和边缘分布的概念，掌握通过样本数据可视化展示这些分布的方法。")
        
        st.subheader("实验内容")
        st.write("对于二维连续型随机变量(X,Y)，本实验通过生成服从特定分布的样本数据，绘制散点图展示联合分布，并通过直方图展示X和Y的边缘分布。")
        
        # 选择分布类型
        col1, col2 = st.columns(2)
        with col1:
            dist_type_x = st.selectbox("X的分布类型", ["正态分布", "均匀分布", "指数分布"])
        with col2:
            dist_type_y = st.selectbox("Y的分布类型", ["正态分布", "均匀分布", "指数分布"])
        
        # 设置相关系数（仅对正态分布有效）
        corr_coef = 0.0
        if dist_type_x == "正态分布" and dist_type_y == "正态分布":
            corr_coef = st.slider("相关系数 ρ", -0.9, 0.9, 0.0)
        
        if st.button("生成联合分布", key="generate_joint"):
            operation_id = record_operation_start(st.session_state.username, "生成二维连续分布")
            # 生成样本数据
            n = 1000
            
            # 根据选择的分布生成数据
            if dist_type_x == "正态分布":
                if dist_type_y == "正态分布" and corr_coef != 0:
                    # 生成相关的二元正态分布
                    mean = [0, 0]
                    cov = [[1, corr_coef], [corr_coef, 1]]  # 协方差矩阵
                    x, y = np.random.multivariate_normal(mean, cov, n).T
                else:
                    x = np.random.normal(0, 1, n)
            elif dist_type_x == "均匀分布":
                x = np.random.uniform(-2, 2, n)
            else:  # 指数分布
                x = np.random.exponential(1, n)
            
            if dist_type_y == "正态分布" and not (dist_type_x == "正态分布" and corr_coef != 0):
                y = np.random.normal(0, 1, n)
            elif dist_type_y == "均匀分布":
                y = np.random.uniform(-2, 2, n)
            else:  # 指数分布
                y = np.random.exponential(1, n)
            
            st.subheader("实验结果")
            # 绘制散点图（联合分布）
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(x, y, alpha=0.5)
            ax1.set_title(f"联合分布散点图 (相关系数: {corr_coef:.2f})")
            ax1.set_xlabel("X值")
            ax1.set_ylabel("Y值")
            st.pyplot(fig1)
            
            # 绘制边缘分布
            col1, col2 = st.columns(2)
            with col1:
                fig2, ax2 = plt.subplots()
                ax2.hist(x, bins=30, color="skyblue", density=True)
                ax2.set_title(f"X的边缘分布 ({dist_type_x})")
                ax2.set_xlabel("X值")
                ax2.set_ylabel("密度")
                st.pyplot(fig2)
            
            with col2:
                fig3, ax3 = plt.subplots()
                ax3.hist(y, bins=30, color="lightgreen", density=True)
                ax3.set_title(f"Y的边缘分布 ({dist_type_y})")
                ax3.set_xlabel("Y值")
                ax3.set_ylabel("密度")
                st.pyplot(fig3)
            
            # 计算并显示相关系数
            if n > 1:
                sample_corr = np.corrcoef(x, y)[0, 1]
                st.subheader(f"样本相关系数: {sample_corr:.4f}")
            
            st.subheader("实验分析")
            st.write("散点图展示了二维随机变量(X,Y)的联合分布情况，可以直观地看出X和Y的取值关系。")
            st.write("X的边缘分布是通过忽略Y值，仅观察X的取值分布得到的，Y的边缘分布同理。")
            st.write("本实验中，X服从%s，Y服从%s，它们的边缘分布分别呈现出相应的分布特征。" % (dist_type_x, dist_type_y))
            
            record_operation_end(operation_id)

# 第四章内容：随机变量的数字特征
def chapter4():
    st.header("第四章 随机变量的数字特征")
    
    # 随机变量的数字特征计算
    with st.expander("数字特征计算", expanded=True):
        st.subheader("实验目的")
        st.write("理解随机变量的数字特征（均值、方差、标准差、偏度、峰度）的概念和计算方法，掌握不同分布的数字特征特点。")
        
        st.subheader("实验内容")
        st.write("对于选定的概率分布，生成样本数据，计算样本的数字特征（均值、方差、标准差、偏度、峰度），并与理论值进行比较，验证样本估计的准确性。")
        
        # 选择分布类型
        dist_type = st.selectbox("分布类型", ["正态分布", "均匀分布", "泊松分布", "指数分布", "二项分布"])
        
        # 根据分布类型显示不同的参数设置
        if dist_type == "正态分布":
            col1, col2 = st.columns(2)
            with col1:
                mean = st.slider("均值μ", -5.0, 5.0, 0.0)
            with col2:
                std = st.slider("标准差σ", 0.1, 3.0, 1.0)
        
        elif dist_type == "均匀分布":
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.slider("最小值", -5.0, 0.0, 0.0)
            with col2:
                max_val = st.slider("最大值", 0.1, 5.0, 1.0)
        
        elif dist_type == "泊松分布":
            lam = st.slider("λ值", 0.1, 20.0, 5.0)
        
        elif dist_type == "指数分布":
            rate = st.slider("率参数λ", 0.1, 2.0, 1.0)
        
        else:  # 二项分布
            col1, col2 = st.columns(2)
            with col1:
                n = st.slider("试验次数n", 1, 100, 10,key="binom_n")
            with col2:
                p = st.slider("成功概率p", 0.01, 0.99, 0.5,key="bernoulli_p")
        
        if st.button("计算数字特征", key="calc_features"):
            operation_id = record_operation_start(st.session_state.username, "计算随机变量数字特征")
            # 生成样本数据
            n_samples = 10000
            
            if dist_type == "正态分布":
                data = np.random.normal(mean, std, n_samples)
                theoretical_mean = mean
                theoretical_var = std ** 2
                theoretical_std = std
                theoretical_skew = 0
                theoretical_kurt = 0  # 正态分布超额峰度为0
            
            elif dist_type == "均匀分布":
                data = np.random.uniform(min_val, max_val, n_samples)
                theoretical_mean = (min_val + max_val) / 2
                theoretical_var = (max_val - min_val) ** 2 / 12
                theoretical_std = np.sqrt(theoretical_var)
                theoretical_skew = 0
                theoretical_kurt = -1.2  # 均匀分布超额峰度
            
            elif dist_type == "泊松分布":
                data = np.random.poisson(lam, n_samples)
                theoretical_mean = lam
                theoretical_var = lam
                theoretical_std = np.sqrt(lam)
                theoretical_skew = 1 / np.sqrt(lam)
                theoretical_kurt = 1 / lam  # 泊松分布超额峰度
            
            elif dist_type == "指数分布":
                data = np.random.exponential(1/rate, n_samples)
                theoretical_mean = 1 / rate
                theoretical_var = 1 / (rate ** 2)
                theoretical_std = 1 / rate
                theoretical_skew = 2
                theoretical_kurt = 6  # 指数分布超额峰度
            
            else:  # 二项分布
                data = np.random.binomial(n, p, n_samples)
                theoretical_mean = n * p
                theoretical_var = n * p * (1 - p)
                theoretical_std = np.sqrt(theoretical_var)
                theoretical_skew = (1 - 2*p) / np.sqrt(n * p * (1 - p)) if (n * p * (1 - p)) > 0 else 0
                theoretical_kurt = (1 - 6*p*(1-p)) / (n * p * (1 - p)) if (n * p * (1 - p)) > 0 else 0  # 二项分布超额峰度
            
            # 计算样本统计量
            sample_mean = np.mean(data)
            sample_var = np.var(data, ddof=1)
            sample_std = np.std(data, ddof=1)
            sample_skew = stats.skew(data)
            sample_kurt = stats.kurtosis(data)  # 超额峰度
            
            st.subheader("实验结果")
            # 绘制分布图
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
            ax.set_title(f"{dist_type} 分布")
            st.pyplot(fig)
            
            # 显示数字特征比较
            st.subheader("理论值与样本估计值比较")
            df = pd.DataFrame({
                "特征": ["均值 (μ)", "方差 (σ²)", "标准差 (σ)", "偏度", "峰度"],
                "理论值": [
                    f"{theoretical_mean:.4f}",
                    f"{theoretical_var:.4f}",
                    f"{theoretical_std:.4f}",
                    f"{theoretical_skew:.4f}",
                    f"{theoretical_kurt:.4f}"
                ],
                "样本估计值": [
                    f"{sample_mean:.4f}",
                    f"{sample_var:.4f}",
                    f"{sample_std:.4f}",
                    f"{sample_skew:.4f}",
                    f"{sample_kurt:.4f}"
                ]
            })
            st.dataframe(df)
            
            st.subheader("实验分析")
            st.write(f"{dist_type}的数字特征描述了该分布的中心位置、离散程度和形状特征：")
            st.write("- 均值表示分布的中心位置")
            st.write("- 方差和标准差表示分布的离散程度")
            st.write("- 偏度表示分布的不对称程度（0表示对称）")
            st.write("- 峰度表示分布的陡峭程度（正态分布峰度为0）")
            st.write("从结果可以看出，随着样本量增大，样本估计值越来越接近理论值，验证了大数定律。")
            
            record_operation_end(operation_id)

# 第五章内容：中心极限定理演示
def chapter5():
    st.header("第五章 中心极限定理演示")
    
    # 中心极限定理
    with st.expander("中心极限定理", expanded=True):
        st.subheader("实验目的")
        st.write("理解中心极限定理的基本思想：无论总体服从什么分布，当样本量足够大时，样本均值的抽样分布近似服从正态分布。")
        
        st.subheader("实验内容")
        st.write("从选定的总体分布中重复抽取一定大小的样本，计算每个样本的均值，观察这些样本均值的分布形态，验证随着样本量增大，样本均值分布是否趋近于正态分布。")
        
        # 选择总体分布类型
        dist_type = st.selectbox("总体分布类型", ["均匀分布", "二项分布", "泊松分布", "指数分布"])
        n_samples = st.slider("样本量", 10, 1000, 30)
        n_trials = st.slider("试验次数", 100, 10000, 1000)
        
        if st.button("执行中心极限定理实验", key="clt_experiment"):
            operation_id = record_operation_start(st.session_state.username, "执行中心极限定理实验")
            # 生成样本均值
            sample_means = []
            
            for _ in range(n_trials):
                if dist_type == "均匀分布":
                    sample = np.random.uniform(0, 1, n_samples)
                elif dist_type == "二项分布":
                    sample = np.random.binomial(10, 0.2, n_samples)  # 偏态二项分布
                elif dist_type == "泊松分布":
                    sample = np.random.poisson(2, n_samples)  # 偏态泊松分布
                else:  # 指数分布
                    sample = np.random.exponential(1, n_samples)
                
                sample_means.append(np.mean(sample))
            
            st.subheader("实验结果")
            # 绘制样本均值分布
            fig, ax = plt.subplots()
            ax.hist(sample_means, bins=30, density=True, alpha=0.6, color='blue')
            
            # 绘制理论正态分布曲线
            mean = np.mean(sample_means)
            std = np.std(sample_means)
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
            
            ax.set_title(f"样本均值分布 (总体: {dist_type}, 样本量: {n_samples})")
            ax.set_xlabel("样本均值")
            ax.set_ylabel("密度")
            st.pyplot(fig)
            
            # 显示统计信息
            st.subheader("样本均值分布统计量")
            st.markdown(f"均值: {mean:.4f}")
            st.markdown(f"标准差: {std:.4f}")
            
            st.subheader("实验分析")
            st.write(f"本实验中，总体服从{dist_type}，这是一个非正态分布。")
            st.write(f"当从该总体中抽取{n_samples}个样本并计算均值，重复{n_trials}次后，样本均值的分布近似服从正态分布。")
            st.write("这验证了中心极限定理：无论总体分布如何，只要样本量足够大，样本均值的抽样分布就近似于正态分布。")
            st.write("中心极限定理是许多统计推断方法的理论基础。")
            
            record_operation_end(operation_id)

    # 大数定律
    with st.expander("大数定律"):
        st.subheader("实验目的")
        st.write("理解大数定律的基本思想：当试验次数足够多时，样本均值会稳定在总体的期望附近。")
        
        st.subheader("实验内容")
        st.write("从选定的总体分布中抽取样本，随着样本量逐渐增加，计算累积样本均值，观察样本均值是否会逐渐收敛于总体的理论期望。")
        
        # 选择分布类型
        dist_type = st.selectbox("分布类型", ["均匀分布", "二项分布", "泊松分布"], key="lln_dist")
        num_trials = st.slider("试验次数", 10, 1000, 100, key="lln_trials")
        
        if st.button("执行大数定律实验", key="lln_experiment"):
            operation_id = record_operation_start(st.session_state.username, "执行大数定律实验")
            # 计算累积均值
            means = []
            errors = []
            
            for i in range(1, num_trials+1):
                if dist_type == "均匀分布":
                    data = np.random.uniform(0, 1, i)
                    expected_mean = 0.5  # 均匀分布[0,1]的期望
                elif dist_type == "二项分布":
                    data = np.random.binomial(10, 0.5, i)
                    expected_mean = 5  # 二项分布(10,0.5)的期望
                else:  # 泊松分布
                    data = np.random.poisson(5, i)
                    expected_mean = 5  # 泊松分布λ=5的期望
                
                current_mean = np.mean(data)
                means.append(current_mean)
                errors.append(abs(current_mean - expected_mean))
            
            st.subheader("实验结果")
            # 创建双轴图表
            fig, ax1 = plt.subplots()
            
            color = 'tab:blue'
            ax1.set_xlabel('试验次数')
            ax1.set_ylabel('样本均值', color=color)
            ax1.plot(range(1, num_trials+1), means, color=color)
            ax1.axhline(y=expected_mean, color='r', linestyle='--', label="期望值")
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
            color = 'tab:red'
            ax2.set_ylabel('估计误差', color=color)
            ax2.plot(range(1, num_trials+1), errors, color=color, alpha=0.5)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()  # 调整布局
            plt.title(f"大数定律演示 (分布: {dist_type})")
            st.pyplot(fig)
            
            # 显示结果
            st.subheader("结果比较")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("理论期望值", expected_mean)
            with col2:
                st.metric("最终样本均值", f"{means[-1]:.4f}")
            
            st.subheader("实验分析")
            st.write(f"本实验中，总体服从{dist_type}，其理论期望为{expected_mean}。")
            st.write(f"随着试验次数从1增加到{num_trials}，样本均值逐渐收敛于理论期望值，估计误差逐渐减小。")
            st.write("这验证了大数定律：当试验次数足够多时，样本均值会稳定在总体的期望附近。")
            st.write("大数定律为用样本均值估计总体期望提供了理论依据。")
            
            record_operation_end(operation_id)

# 第六章内容：统计量与抽样分布
def chapter6():
    st.header("第六章 统计量与抽样分布")
    
    # 样本均值和方差的抽样分布
    with st.expander("样本均值和方差的抽样分布", expanded=True):
        st.subheader("实验目的")
        st.write("理解样本均值和样本方差的抽样分布特性，掌握它们的期望和方差与总体参数的关系。")
        
        st.subheader("实验内容")
        st.write("从正态总体中重复抽取样本，计算每个样本的均值和方差，观察这些统计量的分布形态，并比较它们的期望与理论值。")
        
        # 参数设置
        mu = st.slider("总体均值μ", -5.0, 5.0, 0.0)
        sigma = st.slider("总体标准差σ", 0.1, 3.0, 1.0)
        n = st.slider("样本量", 5, 100, 30)
        
        if st.button("生成抽样分布", key="sampling_dist"):
            operation_id = record_operation_start(st.session_state.username, "生成抽样分布")
            num_simulations = 1000  # 模拟次数
            
            # 生成样本均值和方差
            sample_means = []
            sample_vars = []
            
            for _ in range(num_simulations):
                sample = np.random.normal(mu, sigma, n)
                sample_means.append(np.mean(sample))
                sample_vars.append(np.var(sample, ddof=1))  # 样本方差
            
            st.subheader("实验结果")
            # 绘制样本均值分布
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.hist(sample_means, bins=30, alpha=0.7, color='blue')
                ax1.axvline(x=np.mean(sample_means), color='red', linestyle='dashed', label=f"均值: {np.mean(sample_means):.4f}")
                ax1.set_title("样本均值的分布")
                ax1.set_xlabel("样本均值")
                ax1.set_ylabel("频率")
                ax1.legend()
                st.pyplot(fig1)
            
            # 绘制样本方差分布
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.hist(sample_vars, bins=30, alpha=0.7, color='green')
                ax2.axvline(x=np.mean(sample_vars), color='red', linestyle='dashed', label=f"均值: {np.mean(sample_vars):.4f}")
                ax2.set_title("样本方差的分布")
                ax2.set_xlabel("样本方差")
                ax2.set_ylabel("频率")
                ax2.legend()
                st.pyplot(fig2)
            
            # 显示理论值与实际值比较
            st.subheader("理论值与实际值比较")
            df = pd.DataFrame({
                "统计量": ["样本均值的期望", "样本均值的方差", "样本方差的期望"],
                "理论值": [
                    f"{mu:.4f}",
                    f"{sigma**2/n:.4f}",
                    f"{sigma**2:.4f}"
                ],
                "实际值": [
                    f"{np.mean(sample_means):.4f}",
                    f"{np.var(sample_means):.4f}",
                    f"{np.mean(sample_vars):.4f}"
                ]
            })
            st.dataframe(df)
            
            st.subheader("实验分析")
            st.write(f"从正态总体N(μ={mu}, σ={sigma})中抽取样本量为{n}的样本：")
            st.write("1. 样本均值的抽样分布服从正态分布，其期望等于总体均值μ，方差等于σ²/n")
            st.write("2. 样本方差的抽样分布服从卡方分布，其期望等于总体方差σ²")
            st.write("实验结果验证了这些理论性质，样本估计值与理论值非常接近。")
            
            record_operation_end(operation_id)

    # 次序统计量的分布
    with st.expander("次序统计量的分布"):
        st.subheader("实验目的")
        st.write("理解次序统计量（如最小值、最大值、中位数）的概念和分布特性，掌握它们在描述数据分布中的作用。")
        
        st.subheader("实验内容")
        st.write("从选定的总体分布中重复抽取样本，计算每个样本的最小值、最大值和中位数等次序统计量，观察这些统计量的分布形态。")
        
        # 参数设置
        dist_type = st.selectbox("总体分布类型", ["正态分布", "均匀分布", "指数分布"], key="order_dist")
        sample_size = st.slider("样本量", 5, 100, 20)
        
        if st.button("生成次序统计量", key="order_stats"):
            operation_id = record_operation_start(st.session_state.username, "生成次序统计量")
            num_samples = 1000  # 样本数量
            
            # 存储次序统计量
            mins = []
            maxs = []
            medians = []
            
            for _ in range(num_samples):
                # 生成样本
                if dist_type == "正态分布":
                    sample = np.random.normal(0, 1, sample_size)
                elif dist_type == "均匀分布":
                    sample = np.random.uniform(0, 1, sample_size)
                else:  # 指数分布
                    sample = np.random.exponential(1, sample_size)
                
                # 计算次序统计量
                ordered = np.sort(sample)
                mins.append(ordered[0])
                maxs.append(ordered[-1])
                medians.append(np.median(ordered))
            
            st.subheader("实验结果")
            # 绘制分布图
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.hist(mins, bins=30, alpha=0.7, color='blue')
                ax1.set_title("最小值的分布")
                ax1.set_xlabel("最小值")
                ax1.set_ylabel("频率")
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.hist(medians, bins=30, alpha=0.7, color='green')
                ax2.set_title("中位数的分布")
                ax2.set_xlabel("中位数")
                ax2.set_ylabel("频率")
                st.pyplot(fig2)
            
            with col3:
                fig3, ax3 = plt.subplots()
                ax3.hist(maxs, bins=30, alpha=0.7, color='red')
                ax3.set_title("最大值的分布")
                ax3.set_xlabel("最大值")
                ax3.set_ylabel("频率")
                st.pyplot(fig3)
            
            # 统计量的统计特征
            stats_df = pd.DataFrame({
                "统计量": ["最小值", "中位数", "最大值"],
                "均值": [f"{np.mean(mins):.4f}", f"{np.mean(medians):.4f}", f"{np.mean(maxs):.4f}"],
                "标准差": [f"{np.std(mins):.4f}", f"{np.std(medians):.4f}", f"{np.std(maxs):.4f}"]
            })
            st.subheader("次序统计量的特征")
            st.dataframe(stats_df)
            
            st.subheader("实验分析")
            st.write(f"从{dist_type}中抽取样本量为{sample_size}的样本，计算其最小值、最大值和中位数：")
            st.write("1. 最小值的分布呈现向左偏态，其取值范围受总体下界影响")
            st.write("2. 最大值的分布呈现向右偏态，其取值范围受总体上界影响")
            st.write("3. 中位数的分布相对集中，对于对称分布（如正态分布），中位数的分布接近均值的分布")
            st.write("次序统计量在描述数据分布特征、异常值检测等方面有重要应用。")
            
            record_operation_end(operation_id)

    # 常用的抽样分布
    with st.expander("常用的抽样分布"):
        st.subheader("实验目的")
        st.write("理解统计学中常用的抽样分布（t分布、F分布和卡方分布）的概念和形态特征，掌握它们的应用场景。")
        
        st.subheader("实验内容")
        st.write("通过调整自由度参数，观察t分布、F分布和卡方分布的概率密度函数形态变化，理解自由度对这些分布的影响。")
        
        # t分布
        col1, col2, col3 = st.columns(3)
        with col1:
            df_t = st.slider("t分布自由度", 1, 30, 5)
            if st.button("显示t分布", key="t_dist"):
                x = np.linspace(-4, 4, 100)
                y = stats.t.pdf(x, df=df_t)
                y_norm = stats.norm.pdf(x, 0, 1)  # 标准正态分布
                
                fig, ax = plt.subplots()
                ax.plot(x, y, label=f"t分布 (df={df_t})")
                ax.plot(x, y_norm, 'r--', label="标准正态分布")
                ax.set_title(f"t分布与正态分布比较")
                ax.set_xlabel("x")
                ax.set_ylabel("概率密度")
                ax.legend()
                st.pyplot(fig)
                
                st.subheader("t分布分析")
                st.write(f"t分布的自由度为{df_t}，形状类似正态分布，但尾部更厚。")
                st.write("当自由度增大时，t分布逐渐接近标准正态分布。")
                st.write("t分布主要用于小样本均值的假设检验和置信区间估计。")
        
        # F分布
        with col2:
            df1_f = st.slider("F分布自由度1", 1, 30, 5)
            df2_f = st.slider("F分布自由度2", 1, 30, 5)
            if st.button("显示F分布", key="f_dist"):
                x = np.linspace(0, 5, 100)
                y = stats.f.pdf(x, df1_f, df2_f)
                
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.set_title(f"F分布 (df1={df1_f}, df2={df2_f})")
                ax.set_xlabel("x")
                ax.set_ylabel("概率密度")
                st.pyplot(fig)
                
                st.subheader("F分布分析")
                st.write(f"F分布有两个自由度参数df1={df1_f}和df2={df2_f}。")
                st.write("F分布的取值范围为[0, +∞)，呈现右偏态。")
                st.write("F分布主要用于方差分析、回归模型显著性检验等。")
        
        # 卡方分布
        with col3:
            df_chi = st.slider("卡方分布自由度", 1, 30, 5)
            if st.button("显示卡方分布", key="chi2_dist"):
                x = np.linspace(0, 20, 100)
                y = stats.chi2.pdf(x, df=df_chi)
                
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.set_title(f"卡方分布 (自由度 = {df_chi})")
                ax.set_xlabel("x")
                ax.set_ylabel("概率密度")
                st.pyplot(fig)
                
                st.subheader("卡方分布分析")
                st.write(f"卡方分布的自由度为{df_chi}，取值范围为[0, +∞)。")
                st.write("当自由度增大时，卡方分布逐渐接近正态分布。")
                st.write("卡方分布主要用于拟合优度检验、独立性检验和方差的假设检验等。")

# 第七章内容：点估计
def chapter7():
    st.header("第七章 点估计")
    
    # 矩估计
    with st.expander("矩估计", expanded=True):
        st.subheader("实验目的")
        st.write("理解矩估计的基本思想和方法，掌握利用样本矩估计总体参数的过程，评价矩估计的效果。")
        
        st.subheader("实验内容")
        st.write("从均匀分布总体中抽取样本，利用矩估计方法估计总体参数（下限a和上限b），比较估计值与真实值的差异，观察样本量对估计精度的影响。")
        
        n = st.slider("样本量", 10, 1000, 100)
        
        if st.button("执行矩估计", key="moment_est"):
            operation_id = record_operation_start(st.session_state.username, "执行矩估计")
            # 生成均匀分布的样本数据 (a=0, b=1)
            a_true = 0
            b_true = 1
            data = np.random.uniform(a_true, b_true, n)
            
            # 计算矩估计量
            mu1 = np.mean(data)  # 一阶矩
            mu2 = np.mean(data**2)  # 二阶矩
            a_hat = mu1 - np.sqrt(3 * (mu2 - mu1**2))
            b_hat = mu1 + np.sqrt(3 * (mu2 - mu1**2))
            
            st.subheader("实验结果")
            # 绘制样本直方图
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, alpha=0.6, color='blue')
            ax.axvline(x=a_hat, color='red', linestyle='dashed', label=f"a估计值: {a_hat:.4f}")
            ax.axvline(x=b_hat, color='green', linestyle='dashed', label=f"b估计值: {b_hat:.4f}")
            ax.axvline(x=a_true, color='black', linestyle='-', alpha=0.3, label=f"真实a: {a_true}")
            ax.axvline(x=b_true, color='black', linestyle='-', alpha=0.3, label=f"真实b: {b_true}")
            ax.set_title(f"均匀分布矩估计 (样本量: {n})")
            ax.set_xlabel("样本值")
            ax.set_ylabel("频率")
            ax.legend()
            st.pyplot(fig)
            
            # 显示估计结果
            st.subheader("矩估计结果")
            df = pd.DataFrame({
                "参数": ["a (下限)", "b (上限)"],
                "真实值": [a_true, b_true],
                "估计值": [f"{a_hat:.4f}", f"{b_hat:.4f}"],
                "误差": [f"{abs(a_hat - a_true):.4f}", f"{abs(b_hat - b_true):.4f}"]
            })
            st.dataframe(df)
            
            st.subheader("实验分析")
            st.write(f"从均匀分布U({a_true}, {b_true})中抽取样本量为{n}的样本，使用矩估计方法估计参数：")
            st.write("矩估计的基本思想是用样本矩估计相应的总体矩，然后通过总体矩与参数的关系求解参数估计值。")
            st.write(f"本实验中，a的估计误差为{abs(a_hat - a_true):.4f}，b的估计误差为{abs(b_hat - b_true):.4f}。")
            st.write("随着样本量增大，估计误差会逐渐减小，估计值会越来越接近真实值。")
            
            record_operation_end(operation_id)

    # 最大似然估计
    with st.expander("最大似然估计"):
        st.subheader("实验目的")
        st.write("理解最大似然估计的基本思想和方法，掌握利用最大似然法估计总体参数的过程，比较其与矩估计的异同。")
        
        st.subheader("实验内容")
        st.write("从正态分布总体中抽取样本，利用最大似然估计方法估计总体均值μ和标准差σ，比较估计值与真实值的差异，观察样本量对估计精度的影响。")
        
        n = st.slider("样本量", 10, 1000, 100, key="mle_n")
        
        if st.button("执行最大似然估计", key="mle_est"):
            operation_id = record_operation_start(st.session_state.username, "执行最大似然估计")
            # 生成正态分布的样本数据
            mu_true = 5
            sigma_true = 2
            data = np.random.normal(mu_true, sigma_true, n)
            
            # 计算最大似然估计量
            mu_hat = np.mean(data)  # 均值的MLE
            sigma_hat_mle = np.sqrt(np.mean((data - mu_hat)**2))  # 标准差的MLE
            sigma_hat_unbiased = np.sqrt(np.var(data, ddof=1))  # 无偏估计
            
            st.subheader("实验结果")
            # 绘制样本直方图
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
            # 绘制估计的正态分布曲线
            x = np.linspace(mu_hat - 3*sigma_hat_mle, mu_hat + 3*sigma_hat_mle, 100)
            ax.plot(x, stats.norm.pdf(x, mu_hat, sigma_hat_mle), 'r-', label=f"MLE: N({mu_hat:.2f}, {sigma_hat_mle:.2f})")
            # 绘制真实的正态分布曲线
            ax.plot(x, stats.norm.pdf(x, mu_true, sigma_true), 'g--', label=f"真实: N({mu_true}, {sigma_true})")
            ax.set_title(f"正态分布最大似然估计 (样本量: {n})")
            ax.set_xlabel("样本值")
            ax.set_ylabel("密度")
            ax.legend()
            st.pyplot(fig)
            
            # 显示估计结果
            st.subheader("最大似然估计结果")
            df = pd.DataFrame({
                "参数": ["均值 (μ)", "标准差MLE (σ)", "标准差无偏估计 (σ)"],
                "真实值": [mu_true, sigma_true, sigma_true],
                "估计值": [f"{mu_hat:.4f}", f"{sigma_hat_mle:.4f}", f"{sigma_hat_unbiased:.4f}"],
                "误差": [f"{abs(mu_hat - mu_true):.4f}", 
                         f"{abs(sigma_hat_mle - sigma_true):.4f}",
                         f"{abs(sigma_hat_unbiased - sigma_true):.4f}"]
            })
            st.dataframe(df)
            
            st.subheader("实验分析")
            st.write(f"从正态分布N(μ={mu_true}, σ={sigma_true})中抽取样本量为{n}的样本，使用最大似然估计：")
            st.write("最大似然估计的基本思想是寻找使样本出现概率最大的参数值。")
            st.write(f"本实验中，均值μ的估计误差为{abs(mu_hat - mu_true):.4f}，标准差σ的MLE估计误差为{abs(sigma_hat_mle - sigma_true):.4f}。")
            st.write("对于正态分布，均值的最大似然估计与矩估计相同，但标准差的最大似然估计是有偏的（偏小），样本标准差是其无偏估计。")
            
            record_operation_end(operation_id)

    # 估计量的有效性
    with st.expander("估计量的有效性"):
        st.subheader("实验目的")
        st.write("理解估计量有效性的概念，掌握评价估计量好坏的标准，验证Cramér-Rao下界作为估计量方差下界的性质。")
        
        st.subheader("实验内容")
        st.write("从选定的总体分布中重复抽取样本，计算参数的估计量，估计该估计量的方差，并与Cramér-Rao下界比较，判断该估计量是否为有效估计量。")
        
        # 参数设置
        dist_type = st.selectbox("分布类型", ["正态分布", "均匀分布", "泊松分布"], key="eff_dist")
        sample_size = st.slider("样本量", 10, 1000, 50, key="eff_n")
        
        if st.button("评估估计量有效性", key="eval_eff"):
            operation_id = record_operation_start(st.session_state.username, "评估估计量有效性")
            num_samples = 500  # 模拟次数
            
            # 生成样本数据并计算估计量
            estimates = []
            crlb_values = []
            
            for _ in range(num_samples):
                if dist_type == "正态分布":
                    # 正态分布 N(5, 2^2)
                    sample = np.random.normal(5, 2, sample_size)
                    estimates.append(np.mean(sample))  # 样本均值估计
                    crlb = 4 / sample_size  # CRLB = σ²/n
                    crlb_values.append(crlb)
                
                elif dist_type == "均匀分布":
                    # 均匀分布 U(0, 10)
                    sample = np.random.uniform(0, 10, sample_size)
                    estimates.append(np.mean(sample))  # 样本均值估计
                    crlb = (10**2 / 12) / sample_size  # CRLB = (b-a)²/(12n)
                    crlb_values.append(crlb)
                
                else:  # 泊松分布
                    # 泊松分布 Poisson(5)
                    sample = np.random.poisson(5, sample_size)
                    estimates.append(np.mean(sample))  # 样本均值估计
                    crlb = 5 / sample_size  # CRLB = λ/n
                    crlb_values.append(crlb)
            
            # 计算估计量的方差
            var_estimate = np.var(estimates)
            avg_crlb = np.mean(crlb_values)
            
            st.subheader("实验结果")
            # 绘制估计量分布
            fig, ax = plt.subplots()
            ax.hist(estimates, bins=30, alpha=0.6, color='blue')
            ax.set_title(f"{dist_type} 估计量分布")
            ax.set_xlabel("估计值")
            ax.set_ylabel("频率")
            st.pyplot(fig)
            
            # 显示有效性比较
            st.subheader("有效性比较")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("估计量方差", f"{var_estimate:.6f}")
            with col2:
                st.metric("Cramér-Rao下界平均值", f"{avg_crlb:.6f}")
            
            # 判断是否有效
            if var_estimate < avg_crlb * 1.05:  # 允许微小数值误差
                st.success("估计量达到Cramér-Rao下界，是有效估计量")
            else:
                st.info("估计量未达到Cramér-Rao下界，不是有效估计量")
            
            st.subheader("实验分析")
            st.write(f"从{dist_type}中抽取样本量为{sample_size}的样本，评估样本均值作为估计量的有效性：")
            st.write("有效性是指在所有无偏估计量中，方差最小的估计量称为有效估计量。")
            st.write(f"Cramér-Rao下界提供了一个估计量方差的理论下限，本实验中该下限平均值为{avg_crlb:.6f}。")
            st.write(f"样本均值估计量的实际方差为{var_estimate:.6f}，{'' if var_estimate < avg_crlb * 1.05 else '未'}达到理论下限。")
            st.write("对于本实验中的分布，样本均值通常是总体期望的有效估计量。")
            
            record_operation_end(operation_id)

# 第八章内容：区间估计
def chapter8():
    st.header("第八章 区间估计")
    
    # 置信区间估计
    with st.expander("置信区间估计", expanded=True):
        st.subheader("实验目的")
        st.write("理解置信区间的概念和意义，掌握均值的置信区间估计方法，理解置信水平对区间宽度的影响。")
        
        st.subheader("实验内容")
        st.write("从正态总体中抽取样本，在给定的置信水平下，计算总体均值的置信区间，观察置信区间是否包含总体真实均值，理解置信区间的统计含义。")
        
        # 参数设置
        sample_size = st.slider("样本量", 10, 1000, 50)
        confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95)
        
        if st.button("计算置信区间", key="calc_ci"):
            operation_id = record_operation_start(st.session_state.username, "计算置信区间")
            # 生成正态分布的样本数据
            population_mean = 50
            population_std = 10
            sample = np.random.normal(population_mean, population_std, sample_size)
            
            # 计算样本统计量
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            
            # 计算置信区间
            alpha = 1 - confidence_level
            t_value = stats.t.ppf(1 - alpha/2, df=sample_size - 1)
            margin_error = t_value * sample_std / np.sqrt(sample_size)
            ci_lower = sample_mean - margin_error
            ci_upper = sample_mean + margin_error
            
            st.subheader("实验结果")
            # 绘制样本数据直方图
            fig, ax = plt.subplots()
            ax.hist(sample, bins=30, alpha=0.6, color='blue')
            ax.axvline(x=sample_mean, color='red', linestyle='-', label=f"样本均值: {sample_mean:.2f}")
            ax.axvline(x=ci_lower, color='green', linestyle='--', label=f"{confidence_level*100}% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
            ax.axvline(x=ci_upper, color='green', linestyle='--')
            ax.axvline(x=population_mean, color='black', linestyle='-.', label=f"总体均值: {population_mean}")
            ax.set_title("样本数据分布")
            ax.set_xlabel("样本值")
            ax.set_ylabel("频率")
            ax.legend()
            st.pyplot(fig)
            
            # 显示置信区间结果
            st.subheader("置信区间估计结果")
            st.markdown(f"样本均值: {sample_mean:.4f}")
            st.markdown(f"样本标准差: {sample_std:.4f}")
            st.markdown(f"{confidence_level*100}% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
            st.markdown(f"边际误差: {margin_error:.4f}")
            
            # 判断是否包含真实均值
            if ci_lower <= population_mean <= ci_upper:
                st.success("该置信区间包含总体真实均值")
            else:
                st.warning("该置信区间不包含总体真实均值")
            
            st.subheader("实验分析")
            st.write(f"从均值为{population_mean}的正态总体中抽取样本量为{sample_size}的样本：")
            st.write(f"{confidence_level*100}%置信区间表示，在多次重复抽样中，大约有{confidence_level*100}%的置信区间会包含总体真实均值。")
            st.write(f"本次计算的置信区间为[{ci_lower:.4f}, {ci_upper:.4f}]，{'包含' if ci_lower <= population_mean <= ci_upper else '不包含'}总体真实均值。")
            st.write("置信水平越高，置信区间越宽；样本量越大，置信区间越窄，估计精度越高。")
            
            record_operation_end(operation_id)

    # 正态总体参数的置信区间
    with st.expander("正态总体参数的置信区间"):
        st.subheader("实验目的")
        st.write("掌握正态总体均值和方差的置信区间估计方法，理解不同参数的置信区间构造原理的差异。")
        
        st.subheader("实验内容")
        st.write("从正态总体中抽取样本，分别计算总体均值和总体方差的置信区间，比较这两种置信区间的构造方法和特点。")
        
        # 参数设置
        pop_mean = st.slider("总体均值", -10.0, 10.0, 0.0)
        pop_std = st.slider("总体标准差", 0.1, 5.0, 1.0)
        sample_size = st.slider("样本量", 10, 1000, 50, key="norm_sample_size")
        confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95, key="norm_conf_level")
        
        if st.button("生成置信区间", key="norm_ci"):
            operation_id = record_operation_start(st.session_state.username, "生成正态总体置信区间")
            # 生成样本数据
            sample = np.random.normal(pop_mean, pop_std, sample_size)
            
            # 计算样本统计量
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            
            # 计算总体均值的置信区间
            alpha = 1 - confidence_level
            t_value = stats.t.ppf(1 - alpha/2, df=sample_size - 1)
            margin_error_mean = t_value * sample_std / np.sqrt(sample_size)
            ci_mean = [sample_mean - margin_error_mean, sample_mean + margin_error_mean]
            
            # 计算总体方差的置信区间
            chi2_lower = stats.chi2.ppf(alpha/2, df=sample_size - 1)
            chi2_upper = stats.chi2.ppf(1 - alpha/2, df=sample_size - 1)
            ci_var = [
                (sample_size - 1) * sample_std**2 / chi2_upper,
                (sample_size - 1) * sample_std**2 / chi2_lower
            ]
            
            st.subheader("实验结果")
            # 绘制置信区间
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.errorbar(x=0, y=sample_mean, yerr=margin_error_mean, fmt='bo', capsize=5, label="样本均值")
                ax1.axhline(y=pop_mean, color='r', linestyle='--', label="真实均值")
                ax1.set_title(f"均值的{confidence_level*100}%置信区间")
                ax1.set_xlim(-0.5, 0.5)
                ax1.set_xticks([])
                ax1.legend()
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots()
                yerr = [[sample_std**2 - ci_var[0]],
                            [ci_var[1] - sample_std**2]]
                ax2.errorbar(x=0, y=sample_std**2,
                                      yerr=yerr,
                                      fmt='go', capsize=5, label="样本方差")
                ax2.axhline(y=pop_std**2, color='r', linestyle='--', label="真实方差")
                ax2.set_title(f"方差的{confidence_level*100}%置信区间")
                ax2.set_xlim(-0.5, 0.5)
                ax2.set_xticks([])
                ax2.legend()
                st.pyplot(fig2)

            
            # 显示结果
            st.subheader("置信区间估计结果")
            df = pd.DataFrame({
                "参数": ["总体均值 μ", "总体方差 σ²"],
                "样本估计值": [f"{sample_mean:.4f}", f"{sample_std**2:.4f}"],
                f"{confidence_level*100}% 置信区间": [
                    f"[{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]",
                    f"[{ci_var[0]:.4f}, {ci_var[1]:.4f}]"
                ],
                "是否包含真实值": [
                    "是" if ci_mean[0] <= pop_mean <= ci_mean[1] else "否",
                    "是" if ci_var[0] <= pop_std**2 <= ci_var[1] else "否"
                ]
            })
            st.dataframe(df)
            
            st.subheader("实验分析")
            st.write(f"从正态总体N(μ={pop_mean}, σ={pop_std})中抽取样本量为{sample_size}的样本：")
            st.write("1. 总体均值的置信区间基于t分布构造，当总体标准差未知时使用样本标准差估计。")
            st.write("2. 总体方差的置信区间基于卡方分布构造，与均值的置信区间构造方法不同。")
            st.write(f"本次实验中，均值的{confidence_level*100}%置信区间{'包含' if ci_mean[0] <= pop_mean <= ci_mean[1] else '不包含'}真实均值，")
            st.write(f"方差的{confidence_level*100}%置信区间{'包含' if ci_var[0] <= pop_std**2 <= ci_var[1] else '不包含'}真实方差。")
            
            record_operation_end(operation_id)

# 第九章内容：假设检验
def chapter9():
    st.header("第九章 假设检验")
    
    # 单样本t检验
    with st.expander("单样本t检验", expanded=True):
        st.subheader("实验目的")
        st.write("理解单样本t检验的基本思想和步骤，掌握如何检验样本均值与假设均值之间是否存在显著差异。")
        
        st.subheader("实验内容")
        st.write("从正态总体中抽取样本，设定原假设H₀: μ=μ₀和备择假设H₁: μ≠μ₀，计算t统计量和P值，根据显著性水平做出是否拒绝原假设的决策。")
        
        # 参数设置
        sample_size = st.slider("样本量", 10, 1000, 50)
        hypothesized_mean = st.slider("假设均值μ₀", -10.0, 10.0, 0.0)
        true_mean = st.slider("真实均值μ", -10.0, 10.0, 0.0, key="true_mean")
        alpha = st.slider("显著性水平α", 0.01, 0.1, 0.05)
        
        if st.button("执行单样本t检验        ", key="one_sample_t_test"):
            operation_id = record_operation_start(st.session_state.username, "执行单样本t检验")
            # 生成样本数据
            population_std = 2.0  # 总体标准差
            sample = np.random.normal(true_mean, population_std, sample_size)
            
            # 计算样本统计量
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            
            # 执行t检验
            t_stat, p_value = stats.ttest_1samp(sample, hypothesized_mean)
            df = sample_size - 1  # 自由度
            
            # 计算临界值和拒绝域
            t_critical = stats.t.ppf(1 - alpha/2, df)
            reject_low = hypothesized_mean - t_critical * (sample_std / np.sqrt(sample_size))
            reject_high = hypothesized_mean + t_critical * (sample_std / np.sqrt(sample_size))
            
            # 做出决策
            decision = "拒绝原假设" if p_value < alpha else "不拒绝原假设"
            
            st.subheader("实验结果")
            # 绘制样本分布和检验结果
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制样本直方图
            ax.hist(sample, bins=30, alpha=0.6, color='blue', density=True, label="样本分布")
            
            # 绘制理论正态曲线
            x = np.linspace(min(sample), max(sample), 100)
            ax.plot(x, stats.norm.pdf(x, sample_mean, sample_std), 'r-', label="样本正态近似")
            
            # 标记假设均值和真实均值
            ax.axvline(x=hypothesized_mean, color='green', linestyle='--', label=f"假设均值 μ₀={hypothesized_mean}")
            ax.axvline(x=true_mean, color='black', linestyle='-.', label=f"真实均值 μ={true_mean}")
            ax.axvline(x=sample_mean, color='red', linestyle='-', label=f"样本均值={sample_mean:.2f}")
            
            # 标记拒绝域
            ax.axvspan(min(sample), reject_low, color='gray', alpha=0.3, label="拒绝域")
            ax.axvspan(reject_high, max(sample), color='gray', alpha=0.3)
            
            ax.set_title(f"单样本t检验结果 (α={alpha})")
            ax.set_xlabel("样本值")
            ax.set_ylabel("密度")
            ax.legend()
            st.pyplot(fig)
            
            # 显示检验统计量
            st.subheader("检验统计量")
            stats_df = pd.DataFrame({
                "统计量": ["t值", "P值", "自由度", "临界值"],
                "数值": [
                    f"{t_stat:.4f}",
                    f"{p_value:.6f}",
                    f"{df}",
                    f"{t_critical:.4f}"
                ]
            })
            st.dataframe(stats_df)
            
            # 显示决策结果
            st.subheader("检验结论")
            st.markdown(f"**{decision}** (α = {alpha})")
            st.markdown(f"原假设 H₀: μ = {hypothesized_mean}")
            st.markdown(f"备择假设 H₁: μ ≠ {hypothesized_mean}")
            
            # 解释结果
            if decision == "拒绝原假设":
                st.success(f"在显著性水平α={alpha}下，有足够证据认为总体均值与{hypothesized_mean}存在显著差异。")
            else:
                st.info(f"在显著性水平α={alpha}下，没有足够证据认为总体均值与{hypothesized_mean}存在显著差异。")
            
            st.subheader("实验分析")
            st.write(f"单样本t检验用于检验总体均值是否等于假设值{hypothesized_mean}：")
            st.write(f"1. 当P值小于显著性水平α={alpha}时，我们拒绝原假设，认为存在显著差异。")
            st.write(f"2. 本次检验的t统计量为{t_stat:.4f}，P值为{p_value:.6f}。")
            st.write(f"3. 真实均值为{true_mean}，{'与假设值存在差异' if true_mean != hypothesized_mean else '与假设值一致'}，检验结果{'正确' if (decision == '拒绝原假设' and true_mean != hypothesized_mean) or (decision == '不拒绝原假设' and true_mean == hypothesized_mean) else '错误'}。")
            
            record_operation_end(operation_id)

    # 两样本t检验
    with st.expander("两样本t检验"):
        st.subheader("实验目的")
        st.write("理解两样本t检验的基本思想和应用场景，掌握如何检验两个独立总体的均值是否存在显著差异。")
        
        st.subheader("实验内容")
        st.write("从两个正态总体中分别抽取样本，设定原假设H₀: μ₁=μ₂和备择假设H₁: μ₁≠μ₂，通过t检验判断两个总体的均值是否存在显著差异。")
        
        # 参数设置
        sample_size1 = st.slider("样本量1", 10, 1000, 50, key="sample1_size")
        sample_size2 = st.slider("样本量2", 10, 1000, 50, key="sample2_size")
        mean1 = st.slider("总体1均值μ₁", -10.0, 10.0, 0.0, key="mean1")
        mean2 = st.slider("总体2均值μ₂", -10.0, 10.0, 2.0, key="mean2")
        std1 = st.slider("总体1标准差σ₁", 0.1, 5.0, 1.0, key="std1")
        std2 = st.slider("总体2标准差σ₂", 0.1, 5.0, 1.0, key="std2")
        alpha = st.slider("显著性水平α", 0.01, 0.1, 0.05, key="two_sample_alpha")
        equal_var = st.checkbox("假设两总体方差相等", value=True)
        
        if st.button("执行两样本t检验", key="two_sample_t_test"):
            operation_id = record_operation_start(st.session_state.username, "执行两样本t检验")
            # 生成两个样本
            sample1 = np.random.normal(mean1, std1, sample_size1)
            sample2 = np.random.normal(mean2, std2, sample_size2)
            
            # 计算样本统计量
            mean1_sample = np.mean(sample1)
            mean2_sample = np.mean(sample2)
            std1_sample = np.std(sample1, ddof=1)
            std2_sample = np.std(sample2, ddof=1)
            
            # 执行t检验
            t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
            
            # 做出决策
            decision = "拒绝原假设" if p_value < alpha else "不拒绝原假设"
            
            st.subheader("实验结果")
            # 绘制两个样本的分布
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制直方图
            ax.hist(sample1, bins=30, alpha=0.5, color='blue', density=True, label="样本1")
            ax.hist(sample2, bins=30, alpha=0.5, color='green', density=True, label="样本2")
            
            # 标记样本均值
            ax.axvline(x=mean1_sample, color='blue', linestyle='-', label=f"样本1均值={mean1_sample:.2f}")
            ax.axvline(x=mean2_sample, color='green', linestyle='-', label=f"样本2均值={mean2_sample:.2f}")
            
            ax.set_title("两样本分布比较")
            ax.set_xlabel("样本值")
            ax.set_ylabel("密度")
            ax.legend()
            st.pyplot(fig)
            
            # 显示检验统计量
            st.subheader("检验统计量")
            stats_df = pd.DataFrame({
                "统计量": ["t值", "P值", "均值差"],
                "数值": [
                    f"{t_stat:.4f}",
                    f"{p_value:.6f}",
                    f"{mean1_sample - mean2_sample:.4f}"
                ]
            })
            st.dataframe(stats_df)
            
            # 显示决策结果
            st.subheader("检验结论")
            st.markdown(f"**{decision}** (α = {alpha})")
            st.markdown(f"原假设 H₀: μ₁ = μ₂")
            st.markdown(f"备择假设 H₁: μ₁ ≠ μ₂")
            
            # 解释结果
            if decision == "拒绝原假设":
                st.success(f"在显著性水平α={alpha}下，有足够证据认为两个总体的均值存在显著差异。")
            else:
                st.info(f"在显著性水平α={alpha}下，没有足够证据认为两个总体的均值存在显著差异。")
            
            st.subheader("实验分析")
            st.write("两样本t检验用于比较两个独立总体的均值是否存在显著差异：")
            st.write(f"1. 本次检验中，总体1均值为{mean1}，总体2均值为{mean2}，{'存在真实差异' if mean1 != mean2 else '不存在真实差异'}。")
            st.write(f"2. 检验结果{'正确' if (decision == '拒绝原假设' and mean1 != mean2) or (decision == '不拒绝原假设' and mean1 == mean2) else '错误'}。")
            st.write(f"3. 当两总体方差{'' if equal_var else '不'}相等时，使用{'合并' if equal_var else '分开'}的方差估计。")
            
            record_operation_end(operation_id)

# 主函数
def main():
    # 初始化数据库
    init_db()
    
    # 初始化会话状态
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'login_time' not in st.session_state:
        st.session_state.login_time = ""
    
    # 登录/注册界面
    if not st.session_state.logged_in:
        st.title("概率统计动态演示平台")
        
        tab1, tab2, tab3 = st.tabs(["登录", "注册", "游客访问"])

        
        with tab1:
            st.subheader("用户登录")
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            
            if st.button("登录", type="primary"):
                if authenticate(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.success("登录成功！")
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
        
        with tab2:
            st.subheader("用户注册")
            new_username = st.text_input("新用户名", key="reg_username")
            new_password = st.text_input("新密码", type="password", key="reg_password")
            
            if st.button("注册", type="primary"):
                if register_user(new_username, new_password):
                    st.success("注册成功，请登录")
                else:
                    st.error("注册失败，用户名可能已存在")
        with tab3:
               if st.button("以游客身份访问"):
                      st.session_state.logged_in = True
                      st.session_state.username = "游客"
                      st.session_state.current_login_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                      st.session_state.last_operation_id = None
                      st.session_state.current_page = None
                      st.rerun()

    
    # 登录后的界面
    else:
        # 侧边栏导航
        st.sidebar.title(f"欢迎，{st.session_state.username}")
        page = st.sidebar.radio("选择页面", ["首页",  
                                            "第一章：随机事件及其概率",
                                            "第二章：随机变量及其分布",
                                            "第三章：多维随机变量及其分布",
                                            "第四章：随机变量的数字特征",
                                            "第五章：中心极限定理演示",
                                            "第六章：统计量与抽样分布",
                                            "第七章：点估计",
                                            "第八章：区间估计",
                                            "第九章：假设检验",
                                            "个人中心"])
        
        if st.sidebar.button("退出登录"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.login_time = ""
            st.rerun()
        
        # 显示对应页面
        if page == "首页":
            main_page()
        elif page == "第一章：随机事件及其概率":
            chapter1()
        elif page == "第二章：随机变量及其分布":
            chapter2()
        elif page == "第三章：多维随机变量及其分布":
            chapter3()
        elif page == "第四章：随机变量的数字特征":
            chapter4()
        elif page == "第五章：中心极限定理演示":
            chapter5()
        elif page == "第六章：统计量与抽样分布":
            chapter6()
        elif page == "第七章：点估计":
            chapter7()
        elif page == "第八章：区间估计":
            chapter8()
        elif page == "第九章：假设检验":
            chapter9()
        elif page == "个人中心":
            user_center()

if __name__ == "__main__":
    main()


