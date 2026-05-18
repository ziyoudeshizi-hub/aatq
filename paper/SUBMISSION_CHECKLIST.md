# arXiv 提交清单

## 提交文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `main.tex` | ✅ 就绪 | 完整论文源文件（475行） |
| `layer_degradation.png` | ✅ 就绪 | Figure 1: 逐层隐藏状态退化曲线 |

> **注意**: `references.bib` 未被 main.tex 使用（论文采用内联 `\begin{thebibliography}` 格式），无需上传。`paper_outline.md` 是早期草稿，无需上传。

## 论文元数据

- **标题**: Why Does Layer-Wise Ternary PTQ Fail on Qwen2.5-7B? A Per-Layer Diagnostic of Error Cascading
- **作者**: Zhenhua Jia (Independent Researcher)
- **邮箱**: ziyoudeshizi@gmail.com
- **日期**: May 2026

## 质检结果

- [x] 无 TODO / placeholder / 未完成标记
- [x] 作者信息正确（Zhenhua Jia, Independent Researcher）
- [x] 全部 9 个 `\cite` 引用均有对应 `\bibitem` 条目
- [x] 全部 `\ref{}` 与 `\label{}` 配对正确
- [x] 图片文件路径与 `\includegraphics` 一致
- [x] natbib 与 thebibliography 格式兼容

## 编译说明

本机未安装 pdflatex。编译方式（任选其一）：

### 方式一：本地编译（需安装 TeX 发行版）
1. 安装 [MiKTeX](https://miktex.org/download) 或 [TeX Live](https://tug.org/texlive/)
2. 运行：
   ```
   cd c:\Users\15397\xiaomao\aatq\paper
   pdflatex main.tex
   pdflatex main.tex
   ```
   （运行两次以解析交叉引用；因为使用内联 thebibliography，无需 bibtex 步骤）

### 方式二：Overleaf 在线编译（推荐）
1. 打开 [Overleaf](https://www.overleaf.com)
2. 新建项目 → 上传 `main.tex` 和 `layer_degradation.png`
3. 点击编译，确认 PDF 正常

### 方式三：arXiv 自动编译
arXiv 会自动编译上传的 .tex 文件，提交后可预览 PDF。

## 推荐 arXiv 分类

- **主分类**: `cs.LG` (Machine Learning) — 论文核心是量化方法的诊断分析
- **次分类**: `cs.CL` (Computation and Language) — 应用对象是大语言模型

## arXiv 提交步骤

1. 登录 [arxiv.org](https://arxiv.org)（如无账号需先注册，首次提交需等待认证）
2. 点击 "Submit" → "New Submission"
3. 选择许可证（推荐 CC BY 4.0 或 arXiv non-exclusive license）
4. 上传文件：
   - `main.tex`（主文件）
   - `layer_degradation.png`（图片）
5. 设置元数据：
   - Title: 如上
   - Authors: Zhenhua Jia
   - Abstract: 复制 main.tex 中 abstract 内容
   - Primary Category: cs.LG
   - Cross-list: cs.CL
6. 预览 PDF，确认格式正确
7. 提交。arXiv 审核通常 1-2 个工作日后上线
