# Git 新功能开发与 PR 合并流程

本文档面向第一次使用 Git 和 GitHub 的开发者，记录本项目开发一个新功能时的标准操作流程。

## 一、先理解完整流程

每个新功能都按照下面的顺序进行：

```text
更新 main
  → 从 main 创建功能分支
  → 编写和测试代码
  → 提交到本地 Git
  → 推送到 GitHub
  → 创建并合并 PR
  → 本地切回并更新 main
```

几个常用概念：

- `main`：项目的稳定主分支，应当保存已经完成并验证过的代码。
- 功能分支：开发某一个功能的独立分支，例如 `feature/save-inference-video`。
- `commit`：把当前选中的修改保存成一次有说明的版本记录。
- `push`：把本地提交上传到 GitHub。
- PR（Pull Request）：请求将功能分支合并进 `main`，同时用于检查修改内容和测试结果。

> 重要原则：不要直接在 `main` 上开发新功能。每个新功能都从最新的 `main` 创建一个新分支。

## 二、开始开发新功能

下面以开发“保存推理视频”功能为例，分支名使用 `feature/save-inference-video`。

### 1. 进入项目目录

在 PowerShell 中执行：

```powershell
cd E:\trainstudy_demo\VisionStudyProject
```

后面的 Git 命令都应当在这个目录中执行。

### 2. 检查当前修改

```powershell
git status
```

如果看到下面的内容，说明没有尚未保存的修改，可以继续：

```text
nothing to commit, working tree clean
```

如果 Git 显示有修改，不要直接切换分支或拉取代码。应当先确认这些修改属于哪个功能，再决定提交或处理。

### 3. 切回并更新 main

```powershell
git switch main
git pull --ff-only origin main
```

- `git switch main`：切换到本地 `main`。
- `git pull --ff-only origin main`：从 GitHub 获取最新的 `main`，并阻止意外产生额外的合并提交。
- 如果显示 `Already up to date.`，表示本地已经是最新版本，不是错误。

### 4. 创建并进入功能分支

```powershell
git switch -c feature/save-inference-video
```

`-c` 表示创建新分支，并立即切换到该分支。可以用下面的命令确认：

```powershell
git branch --show-current
```

预期输出：

```text
feature/save-inference-video
```

分支名建议使用简短的英文，并按用途添加前缀：

- `feature/...`：新功能。
- `fix/...`：修复问题。
- `docs/...`：只修改文档。
- `test/...`：只增加或调整测试。

## 三、开发、检查和提交

### 5. 编写代码并运行测试

在功能分支中修改代码。完成一个阶段后，运行与修改相关的脚本和测试。

本项目的完整单元测试命令为：

```powershell
E:\anaconda\envs\part_yolo_gpu\python.exe -m unittest discover -s tests -v
```

不要只确认代码能够保存，还要检查它是否能够正常运行，以及已有功能是否受到影响。

### 6. 检查修改内容

```powershell
git status
git diff
```

- `git status`：查看新增、修改或删除了哪些文件。
- `git diff`：查看文件中具体修改了哪些内容。

确认没有把模型权重、训练输出、临时文件或其他无关内容加入本次功能。

### 7. 暂存准备提交的文件

建议明确写出要提交的文件，例如：

```powershell
git add config.py scripts/04_realtime_inference.py README.md
```

如果已经通过 `git status` 确认所有修改都属于当前功能，也可以执行：

```powershell
git add .
```

提交前再次检查暂存内容：

```powershell
git diff --staged
```

### 8. 创建本地提交

```powershell
git commit -m "add inference video recording"
```

提交说明应当简短地描述“这次提交完成了什么”。常见示例：

```text
add camera inference script
fix readonly inference frame
update inference documentation
```

一个功能可以包含多个清晰的小提交，不需要把所有工作都塞进一个提交。

## 四、推送并创建 PR

### 9. 推送功能分支到 GitHub

第一次推送该分支时执行：

```powershell
git push -u origin feature/save-inference-video
```

`-u` 会建立本地分支与 GitHub 分支的跟踪关系。此后在同一分支继续提交，只需执行：

```powershell
git push
```

### 10. 创建 PR

初学者可以直接使用 GitHub 网页：

1. 打开本项目的 GitHub 仓库。
2. 点击 `Compare & pull request`。
3. 确认 `base` 是 `main`。
4. 确认 `compare` 是当前功能分支。
5. 填写修改说明和测试结果，然后创建 PR。

也可以使用已经登录的 GitHub CLI：

```powershell
gh pr create --base main --head feature/save-inference-video
```

创建 PR 不等于已经合并。PR 用来确认代码修改、测试结果以及合并目标是否正确。

### 11. 合并 PR

确认功能和测试都没有问题后，在 GitHub PR 页面依次点击：

```text
Merge pull request
Confirm merge
```

合并完成后，GitHub 上的 `main` 已经包含新功能，但本地 `main` 不会自动更新。

## 五、合并后更新本地 main

### 12. 切回并同步 main

```powershell
git switch main
git pull --ff-only origin main
```

验证当前状态：

```powershell
git status -sb
git log -1 --oneline --decorate
```

正常情况下会看到类似内容：

```text
## main...origin/main
093ec87 (HEAD -> main, origin/main) Merge pull request ...
```

其中 `HEAD -> main` 和 `origin/main` 位于同一条提交上，表示本地与 GitHub 已同步。

### 13. 可选：删除已经合并的功能分支

确认 PR 已合并后，可以删除本地功能分支：

```powershell
git branch -d feature/save-inference-video
```

如果还要删除 GitHub 上的远程功能分支：

```powershell
git push origin --delete feature/save-inference-video
```

删除功能分支不是必须操作，也不会删除已经合并进 `main` 的代码。

## 六、继续开发同一个功能

如果功能还没有完成，不要创建新的分支。先确认仍在原功能分支：

```powershell
git branch --show-current
```

继续修改后重复以下步骤：

```powershell
git status
git diff
git add <本次需要提交的文件>
git diff --staged
git commit -m "说明本次修改"
git push
```

## 七、常见情况

### 不小心在 main 上修改了代码，但还没有提交

先不要提交到 `main`，可以直接基于当前状态创建功能分支：

```powershell
git switch -c feature/合适的功能名称
```

未提交的修改会跟随到新分支，然后再正常检查、暂存和提交。

### 切换分支时提示本地修改会被覆盖

停止操作，不要强制切换，也不要使用 `git reset --hard`。先运行：

```powershell
git status
git diff
```

确认修改属于哪个功能后，再决定提交或寻求帮助。

### pull 时显示 Already up to date

这表示当前本地分支已经与 GitHub 同步，不需要额外处理。

### 不确定自己在哪个分支

```powershell
git branch --show-current
```

开始新功能前应当先位于最新的 `main`；开发过程中应当位于对应的功能分支。

## 八、每次开发时的简要清单

开始开发：

```powershell
cd E:\trainstudy_demo\VisionStudyProject
git status
git switch main
git pull --ff-only origin main
git switch -c feature/功能名称
```

完成开发：

```powershell
git status
git diff
git add <需要提交的文件>
git diff --staged
git commit -m "提交说明"
git push -u origin feature/功能名称
```

PR 合并后：

```powershell
git switch main
git pull --ff-only origin main
git status -sb
```
