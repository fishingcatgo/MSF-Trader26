


import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 1. tensorboard 导出csv
def export_to_csv_unique(log_dir, output_dir="csv_exports_unique"):
    """
    将 TensorBoard 事件文件中的标量数据导出为无重复行的 CSV。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                file_path = os.path.join(root, file)
                
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()

                tags = event_acc.Tags().get('scalars', [])
                
                if not tags:
                    print(f"在文件 {file} 中没有找到标量数据。")
                    continue

                print(f"正在处理文件: {file}")
                
                data = {}
                for tag in tags:
                    events = event_acc.Scalars(tag)
                    steps = [event.step for event in events]
                    values = [event.value for event in events]
                    data[tag] = pd.DataFrame({
                        'step': steps,
                        'value': values,
                    })

                # 合并所有数据到一个 DataFrame
                merged_df = pd.DataFrame()
                for tag, df in data.items():
                    df = df.rename(columns={'value': tag})
                    if merged_df.empty:
                        merged_df = df
                    else:
                        merged_df = pd.merge(merged_df, df, on='step', how='outer')
                
                # --- 新增步骤：对 DataFrame 进行去重 ---
                # 根据 'step' 列去除重复行，并保留第一次出现的记录
                merged_df = merged_df.drop_duplicates(subset=['step'], keep='first')
                
                # 创建输出文件名
                output_filename = os.path.join(output_dir, f"{file}.csv")
                
                # 导出到 CSV
                merged_df.to_csv(output_filename, index=False)
                print(f"数据已成功导出到 {output_filename}")



# 2. 取出最优值
def export_to_csv_unique_top(log_dir, output_dir="csv_exports_unique",metrics_col="train/custom_treshold",max_return_col='eval/return_maxrate',
                            best_metrics_col="train/custom_treshold", top_k=5, mode="max"):
    """
    将 TensorBoard 事件文件中的标量数据导出为无重复行的 CSV。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    best_result = { 
                    "extreme": None,
                    "max_return": None,
                    "top_k": None,
                    "average": None
                            }

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                file_path = os.path.join(root, file)
                
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()

                tags = event_acc.Tags().get('scalars', [])
                
                if not tags:
                    print(f"在文件 {file} 中没有找到标量数据。")
                    continue

                print(f"正在处理文件: {file}")
                
                data = {}
                for tag in tags:
                    events = event_acc.Scalars(tag)
                    steps = [event.step for event in events]
                    values = [event.value for event in events]
                    data[tag] = pd.DataFrame({
                        'step': steps,
                        'value': values,
                    })

                # 合并所有数据到一个 DataFrame
                merged_df = pd.DataFrame()
                for tag, df in data.items():
                    df = df.rename(columns={'value': tag})
                    if merged_df.empty:
                        merged_df = df
                    else:
                        merged_df = pd.merge(merged_df, df, on='step', how='outer')
                
                # --- 新增步骤：对 DataFrame 进行去重 ---
                # 根据 'step' 列去除重复行，并保留第一次出现的记录
                merged_df = merged_df.drop_duplicates(subset=['step'], keep='first')
                
                # 创建输出文件名
                output_filename = os.path.join(output_dir, f"{file}.csv")
                
                # 导出到 CSV
                merged_df.to_csv(output_filename, index=False)
                print(f"数据已成功导出到 {output_filename}")

                print(merged_df.columns)

                merged_df.columns = merged_df.columns.str.strip()

                 # 返回最优值
                if metrics_col not in merged_df.columns or best_metrics_col not in merged_df.columns:
                    if best_metrics_col in merged_df.columns:
                        if mode == "max":
                            extreme_val = merged_df[best_metrics_col].max()
                            max_return = merged_df[max_return_col].max() if  max_return_col in merged_df.columns else None

                        else:
                            extreme_val = merged_df[best_metrics_col].min()
                            max_return = merged_df[max_return_col].min() if  max_return_col in merged_df.columns else None

                        best_result ={
                                    "extreme": extreme_val,
                                    "max_return": max_return,
                                    "top_k": None,
                                    "average": None
                                }
                    
                    continue

                if max_return_col not in merged_df.columns:
                    continue


             
                    

                if mode == "max":
                    selected = merged_df.nlargest(top_k, metrics_col)
                    extreme_val = merged_df[best_metrics_col].max()
                    max_return = merged_df[max_return_col].max()
                else:
                    selected = merged_df.nsmallest(top_k, metrics_col)
                    extreme_val = merged_df[best_metrics_col].min()
                    max_return = merged_df[max_return_col].min()


                # print(type(selected))
                avg_val = selected[metrics_col].mean()
                best_result ={
                    "extreme": extreme_val,
                    "max_return": max_return,
                    "top_k": selected[metrics_col].tolist(),
                    "average": avg_val
                }
    return best_result


# 3. 取出所有文件最优值,所有文件结果以list返回
def unique_top_all_result(log_dir, output_dir="csv_exports_unique",metrics_col="train/custom_treshold",max_return_col='eval/return_maxrate',
                            best_metrics_col="train/custom_treshold", top_k=5, mode="max"):
    """
    将 TensorBoard 事件文件中的标量数据导出为无重复行的 CSV。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    best_result = { 
                    "extreme": None,
                    "max_return": None,
                    "top_k": None,
                    "average": None,
                    "file": None
                            }
    all_best=[]
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                file_path = os.path.join(root, file)
                
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()

                tags = event_acc.Tags().get('scalars', [])
                
                if not tags:
                    print(f"在文件 {file} 中没有找到标量数据。")
                    continue

                print(f"正在处理文件: {file}")
                
                data = {}
                for tag in tags:
                    events = event_acc.Scalars(tag)
                    steps = [event.step for event in events]
                    values = [event.value for event in events]
                    data[tag] = pd.DataFrame({
                        'step': steps,
                        'value': values,
                    })

                # 合并所有数据到一个 DataFrame
                merged_df = pd.DataFrame()
                for tag, df in data.items():
                    df = df.rename(columns={'value': tag})
                    if merged_df.empty:
                        merged_df = df
                    else:
                        merged_df = pd.merge(merged_df, df, on='step', how='outer')
                
                # --- 新增步骤：对 DataFrame 进行去重 ---
                # 根据 'step' 列去除重复行，并保留第一次出现的记录
                merged_df = merged_df.drop_duplicates(subset=['step'], keep='first')
                
                # 创建输出文件名
                output_filename = os.path.join(output_dir, f"{file}.csv")
                
                # 导出到 CSV
                merged_df.to_csv(output_filename, index=False)
                print(f"数据已成功导出到 {output_filename}")

                print(merged_df.columns)

                merged_df.columns = merged_df.columns.str.strip()

                 # 返回最优值
                if metrics_col not in merged_df.columns or best_metrics_col not in merged_df.columns:
                    if best_metrics_col in merged_df.columns:
                        if mode == "max":
                            extreme_val = merged_df[best_metrics_col].max()
                            max_return = merged_df[max_return_col].max() if  max_return_col in merged_df.columns else None

                        else:
                            extreme_val = merged_df[best_metrics_col].min()
                            max_return = merged_df[max_return_col].min() if  max_return_col in merged_df.columns else None

                        best_result ={
                                    "extreme": extreme_val,
                                    "max_return": max_return,
                                    "top_k": None,
                                    "average": None,
                                    "file": f"✅file_path:{root},"  # 带文件路径
                                }
                    all_best.append(best_result)
                    continue

                if max_return_col not in merged_df.columns:
                    all_best.append(best_result)
                    continue


                 # if metrics_col not in merged_df.columns or best_metrics_col not in merged_df.columns:
                #     print('所有列名',merged_df.columns)
                #     print('metrics_col：',metrics_col, 'best_metrics_col：',best_metrics_col)
                    
                #     continue
                  
                # if best_metrics_col not in merged_df.columns:
                #     print('所有列名',merged_df.columns)
                #     print('best_metrics_col：',best_metrics_col)
                    
                #     continue
                # if metrics_col not in merged_df.columns:
                #     # print('所有列名',merged_df.columns)
                #     # print('metrics_col：',metrics_col)
                #     print('metrics_col不存在：',metrics_col)

                #     continue
                    

                if mode == "max":
                    selected = merged_df.nlargest(top_k, metrics_col)
                    extreme_val = merged_df[best_metrics_col].max()
                    max_return = merged_df[max_return_col].max()
                else:
                    selected = merged_df.nsmallest(top_k, metrics_col)
                    extreme_val = merged_df[best_metrics_col].min()
                    max_return = merged_df[max_return_col].min()


                # print(type(selected))
                avg_val = selected[metrics_col].mean()
                best_result ={
                    "extreme": extreme_val,
                    "max_return": max_return,
                    "top_k": selected[metrics_col].tolist(),
                    "average": avg_val,
                    "file": f"✅file_path:{root},"  # 带文件路径
                }
                all_best.append(best_result)
    return all_best






import importlib
import json
import os
if __name__ == "__main__":
    
    # 1.取出某个文件最优值
    # resul=export_to_csv_unique_top("./logs_train_testTime/config5_2025-09-17_15-57-38/logs", "./logs_train_testTime/config5_2025-09-17_15-57-38/logs/csv_logs2",metrics_col="train/custom_treshold", best_metrics_col="train/custom_best", top_k=500, mode="min") #./logs/2025-09-06_00-40-59 
    # resul=export_to_csv_unique_top("./logs_train_ppo/config1_2025-09-17_20-12-34/ppo", "./logs_train_ppo/config1_2025-09-17_20-12-34/ppo",
    #                                metrics_col="eval/custom_threshold", best_metrics_col="eval/custom_best", top_k=5, mode="max") #./logs/2025-09-06_00-40-59 
    # print(resul)


    # 2. 所有最优值，一般训练完再执行，查看全部结果
    all_results=unique_top_all_result("./logs_train_ppo_v4_2", "./logs_train_ppo_v4_2/aout_log",
                                   metrics_col="eval/custom_threshold", best_metrics_col="eval/custom_best", top_k=5, mode="max") #./logs/2025-09-06_00-40-59 
    # print(all_results)

    print('all_results:',all_results)
    # 假设是 list[dict]  
    # 保存 JSONL
    LOG_DIR='./logs_train_ppo_v4_2/aout_log/'
    jsonl_path = os.path.join(LOG_DIR, f"global_best_.jsonl")
    # csv_path = os.path.join(LOG_DIR, f"global_best_{timestamp}.csv")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

