\# Day 12：GitHub 远程仓库上传记录



\## 今日目标



本次工作的目标是将本地端侧 AI / ONNX Runtime 部署项目上传到 GitHub，使项目具备在线展示、简历链接引用和后续版本管理能力。



\## 完成内容



\### 1. 检查大模型文件



上传前使用 Git 检查已追踪文件，确认 `models/\*.onnx` 文件没有被加入 Git 版本管理。



这一步是必要的，因为 ONNX 模型文件较大，不适合直接上传到普通 GitHub 仓库中。项目中的 `.gitignore` 已经忽略：



\* models/\*.onnx

\* models/\*.pth

\* models/\*.pt



\### 2. 创建 GitHub 远程仓库



远程仓库地址：



\* https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark



仓库名称：



\* edge-ai-onnxruntime-benchmark



项目定位：



\* Edge AI

\* ONNX Runtime

\* Model Deployment

\* Inference Benchmark

\* Computer Vision



\### 3. 配置远程仓库地址



本地仓库已配置远程地址：



\* origin: https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark.git



\### 4. 推送本地 main 分支



执行命令：



\* git push -u origin main



推送结果：



\* 本地 main 分支已成功推送到 GitHub

\* 本地 main 已设置为追踪 origin/main

\* GitHub 认证通过

\* 远程仓库已成功创建项目内容



\## 当前项目状态



项目目前已经具备：



\* 本地 Git 版本管理

\* GitHub 远程仓库

\* README 首页展示

\* scripts 源代码

\* docs 项目文档

\* results 实验结果

\* 性能曲线图片

\* requirements.txt

\* GitHub 上传说明

\* 简历项目描述

\* 面试介绍稿



\## 当前 Git 提交记录



当前项目已有主要提交：



\* Initial edge AI ONNX Runtime benchmark project

\* Add Day 9 engineering docs

\* Add resume and interview materials

\* Polish README for GitHub presentation



\## 下一步计划



Day 13 建议继续完成：



\* 检查 GitHub 页面展示效果

\* 修复 README 图片路径或显示问题

\* 生成最终简历项目版本

\* 准备投递岗位匹配清单

\* 准备项目 30 秒、1 分钟、3 分钟讲解版本



