\# Day 18：Word / PDF 简历生成记录



\## 今日目标



本次工作的目标是基于一页纸 Markdown 简历，使用 Python 在本地生成 Word 版本简历，并通过 Word 导出 PDF，形成可用于投递的简历文件。



\## 完成内容



\### 1. 新增 Word 简历生成脚本



新增文件：



\- scripts/generate\_resume\_docx.py



该脚本使用 python-docx 生成 Word 简历，包含：



\- 个人信息

\- 教育背景

\- 专业技能

\- 端侧 AI / ONNX Runtime 工程项目

\- 低轨卫星边缘智能科研项目

\- 算法与编程训练

\- 求职定位



\### 2. 生成 Word 简历



生成文件：



\- outputs/端侧AI模型部署方向\_简历\_v1.docx



该文件为端侧 AI / 模型部署方向的一页纸简历初版。



\### 3. 导出 PDF 简历



通过 Microsoft Word 另存为 PDF，生成：



\- outputs/端侧AI模型部署方向\_简历\_v1.pdf



\## 当前状态



截至 Day 18，已经完成：



\- Markdown 简历版本

\- 一页纸简历版本

\- 三方向岗位定制版本

\- Word 简历生成脚本

\- Word 简历初版

\- PDF 简历初版



\## 下一步计划



Day 19 建议继续：



\- 检查 Word / PDF 简历排版

\- 压缩到真正一页

\- 替换个人真实姓名、电话、邮箱

\- 根据端侧 AI / 通信 AI / AI 工程化三个方向分别生成正式投递版

