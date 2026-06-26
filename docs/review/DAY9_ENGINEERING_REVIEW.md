\# Day 9：项目工程化整理记录



\## 今日目标



本次工作的目标是将前 7 天完成的端侧 AI 推理实验项目整理为更适合求职展示、GitHub 上传和面试讲解的工程化项目结构。



\## 完成内容



\### 1. 目录结构整理



新增：



\- configs/

\- docs/

\- docs/daily\_logs/

\- docs/resume/

\- docs/review/



将每日实验记录、简历项目描述和项目检查文档进行分类整理。



\### 2. 新增配置文件



新增：



\- configs/experiment\_config.json



用于记录模型名称、输入尺寸、batch size、ONNX 模型路径、benchmark 输出路径等实验配置。



\### 3. 新增一键运行脚本



新增：



\- scripts/run\_all\_pipeline.py

\- run\_pipeline.bat



可以一键运行 PyTorch 推理、ONNX 导出、ONNX Runtime 推理、batch benchmark 和性能曲线生成流程。



\### 4. 新增 GitHub 上传说明



新增：



\- GITHUB\_UPLOAD\_GUIDE.md



说明哪些文件适合上传，哪些大模型文件不建议上传。



\### 5. 新增面试问答文档



新增：



\- docs/INTERVIEW\_QA.md



整理项目面试中可能被问到的问题，例如为什么使用 ONNX、如何验证导出正确性、batch size 测试说明什么等。



\## 当前项目状态



项目已经具备：



\- 可运行代码

\- 实验结果

\- CSV 性能数据

\- 可视化图表

\- README 文档

\- 简历项目描述

\- 面试问答

\- 一键运行脚本



已经从学习 demo 初步整理成求职展示项目。



\## 下一步计划



Day 10 建议继续完成：



\- 简历项目 bullet 精修

\- 1 分钟项目介绍

\- 3 分钟项目讲解

\- 面试追问回答

\- 投递岗位匹配分析

