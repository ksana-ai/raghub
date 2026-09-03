# RAGHub

[English](README.md) | 简体中文

[![CI](https://github.com/ksana-ai/raghub/actions/workflows/ci.yml/badge.svg)](https://github.com/ksana-ai/raghub/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **实验性预览版本：** RAGHub 适用于本地评测和合成数据实验。它尚未达到生产可用状态，不能直接暴露在不可信网络中。

RAGHub 是一个以证据为基础构建的 Go 原生检索平台，而不是聊天 Demo 的包装层。当前仓库实现了三条可以独立度量的检索基线：

`Markdown 摄取 -> PostgreSQL + pgvector 版本化存储 -> FTS、精确 Dense 或失败关闭的 Hybrid RRF -> 租户/ACL 过滤 -> 引用 -> 三路离线评测`

## 当前状态

本阶段已经实现：

- 确定性的、感知标题层级的 Markdown 分块；
- 不可变文档版本，以及幂等的重复摄取；
- 分离存储的 `raw_text` 和 `indexed_text` 字段；
- 使用 GIN 索引的 PostgreSQL 加权全文检索；
- 真实的批量 OpenAI 兼容 Embedding 客户端，默认连接 LM Studio 的 `text-embedding-bge-m3`，维度为 1024；
- 不可变 Embedding Profile，明确记录 provider、model、dimension 以及文档/查询处理配方的来源；
- 文档版本激活前原子写入向量，并支持对已有 FTS 文档版本进行幂等向量回填；
- 在已物化的“有权访问且为当前版本”的集合上执行精确 pgvector 余弦检索；
- 显式的 `fts`、`dense`、`hybrid` 模式，以及查询向量化、数据库和融合阶段的 trace；
- 在分别完成权限过滤的 FTS 与 Dense 候选集上执行失败关闭的倒数排名融合，预注册候选下限为 20/20，`rrf_k=60`；
- 在 SQL 查询内执行当前版本、租户和 principal ACL 过滤；
- 每条命中均返回精确的文档版本、chunk 和来源引用；
- 每阶段的分数、排名和耗时 trace；
- 版本化的词法、配对和预注册 Hybrid smoke 数据集；
- 冻结的 50-query 困难基准集，包含多 gold、ACL 和租户隔离用例；
- 每次 Hybrid 评测都记录精确的内部 FTS/Dense 候选证据，并检查禁止候选；
- 严格 JSON 评测，以及向后兼容的两路比较、严格三路比较和候选诊断 manifest；
- HitRate@K、标准 Recall@K、MRR、二元相关性 nDCG@K、p50 和 p95；
- 单元测试和可选启用的 PostgreSQL 集成测试。

尚未实现：

- 近似最近邻索引或生产规模性能证明；
- reranking；
- 答案生成、引用验证或 LLM Judge；
- Contextual Retrieval、Agentic RAG、RAPTOR 或 GraphRAG；
- 生产级认证、部署或生产性能证据。

这个区别很重要：当前成果是 FTS、精确 Dense 和 Hybrid RRF 的 smoke 基线，不是生产级 RAG 系统。

## 快速开始

环境要求：

- Go 1.25 或更高版本；module 推荐使用已修复安全问题的 Go 1.27.1 工具链；
- Docker 和 Docker Compose；
- LM Studio 通过 OpenAI 兼容的 `/v1/embeddings` API 提供 `text-embedding-bge-m3`，用于文档摄取、Dense 检索和 Hybrid 检索。

启动 PostgreSQL：

```bash
docker compose up -d postgres
export RAGHUB_DATABASE_URL='postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable'
export RAGHUB_EMBEDDING_ENDPOINT='http://127.0.0.1:1234/v1/embeddings'
```

如果宿主机上的 LM Studio 已经开始监听，也可以同时构建并启动 API 和 PostgreSQL 容器：

```bash
docker compose --profile app up --build
```

容器配置会把 API 映射到 `http://localhost:8080`，并通过 `host.docker.internal` 访问 Embedding 服务。`compose.yaml` 中的开发数据库凭据只适用于本地，不能在共享环境或生产环境中复用。

如果端口 `55432` 已被占用，请为 Compose 设置 `RAGHUB_POSTGRES_PORT`，并在 `RAGHUB_DATABASE_URL` 中使用相同端口。

启动 API 并应用内嵌数据库迁移：

```bash
go run ./cmd/raghub-api -migrate
```

在另一个终端中摄取一份租户级共享的 Markdown 文档：

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{
    "document_id": "deployment-guide",
    "title": "Deployment Guide",
    "source_uri": "https://docs.example.test/deployment",
    "content": "# Deployment\n\nUse blue-green deployment for zero-downtime releases.",
    "metadata": {"team": "platform"}
  }' \
  http://localhost:8080/v1/documents
```

使用词法基线进行检索：

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"blue green deployment","top_k":5,"mode":"fts"}' \
  http://localhost:8080/v1/search
```

将 `mode` 改为 `dense`，运行 Dense 基线：

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"How can releases avoid an outage?","top_k":5,"mode":"dense"}' \
  http://localhost:8080/v1/search
```

使用 `mode=hybrid` 运行固定协议的 Hybrid RRF 基线：

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"How can releases avoid an outage?","top_k":5,"mode":"hybrid"}' \
  http://localhost:8080/v1/search
```

Hybrid 会分别请求 `max(top_k, 20)` 个已完成权限过滤的候选，使用 `k=60` 执行 RRF，并返回 FTS、Dense 和 RRF 的排名与分数。如果任一检索分支失败，整个 Hybrid 请求都会失败，而不会静默改变检索协议。

`X-Tenant-ID` 和 `X-Principal-ID` 只是开发环境的授权上下文。它们虽然被刻意放在 JSON 请求体之外，但在可信认证层根据凭据生成这些值之前，仍是可以伪造的请求头。`/readyz` 同时检查 PostgreSQL 和配置的 Embedding 模型；`/healthz` 只证明进程存活。

## 评测

预注册的 benchmark v1 包含 44 份文档上的 50 条固定查询，其中 8 条查询具有两个 gold。覆盖范围包括精确消歧、语义改写、中英跨语言检索、近重复干扰项、多相关结果召回、ACL 过滤和租户隔离。冻结后的精确字节 SHA-256 为：

`aa44175b9ae656d97473a8340ebac59bc1432d7cee90e51432c2b4f89e61f85f`

提交 benchmark 和 evaluator 变更后，运行完整的 clean-revision 协议：

```bash
make eval-benchmark
```

该命令会生成 FTS、Dense 和 Hybrid 的 Top-5 manifest，并进行严格的三路比较；随后独立运行 Top-20 FTS 与 Dense，生成候选诊断。分类使用每次 Hybrid 请求中实际捕获的、已完成权限过滤的分支候选，Top-20 独立运行作为辅助分支指标。诊断类别如下：

- `fusion_ordering_gap`：所有缺失 gold 都至少被一个分支生成，reranker 可能有帮助；
- `candidate_generation_gap`：缺失 gold 没有进入任何分支候选集，reranker 无法解决；
- `mixed_gap`：只有部分缺失 gold 被分支生成；
- `complete`：Hybrid 在 Top 5 中召回了全部 gold。

候选集只写入评测 manifest，不会出现在公开 API JSON 中。即使禁止候选没有进入最终 Hybrid 结果，只要它出现在内部候选中，评测安全门禁仍会失败。

预注册的 v3 smoke 数据集包含 20 条固定查询，覆盖精确标识符、语义改写、中英跨语言检索、近重复干扰项、ACL 过滤和租户隔离。开发时可在不改变其字节的前提下分别运行：

```bash
mkdir -p artifacts/evals
go run ./cmd/raghub-eval \
  -migrate \
  -mode fts \
  -dataset datasets/smoke/v3.json \
  -output artifacts/evals/v3-fts.json

go run ./cmd/raghub-eval \
  -mode dense \
  -dataset datasets/smoke/v3.json \
  -output artifacts/evals/v3-dense.json

go run ./cmd/raghub-eval \
  -mode hybrid \
  -dataset datasets/smoke/v3.json \
  -output artifacts/evals/v3-hybrid.json
```

单次运行适合作为提交前验收。可信的三路比较还要求所有 manifest 指向同一个干净提交。提交已验收阶段后，可使用失败即停止的目标生成并比较三路结果：

```bash
make eval-three-way
```

该目标会在检索前检查 revision 已提交且工作区干净，然后构建 evaluator，使 Go 将被检查的 VCS revision 嵌入二进制，并把 FTS、Dense、Hybrid 和三路比较结果写入 `artifacts/evals/`。`make eval-v3-regression` 和 `make eval-v2-regression` 会在保留的 smoke 数据集上运行相同协议；`make eval-all` 会依次执行 50-query benchmark 和两个回归集。`make eval-paired` 保留为当前三路协议的兼容别名。

每份 report-v3 manifest 都记录数据集 hash、检索器与配置身份、运行时信息、聚合指标、逐查询排名/分数以及内部候选 ID/排名。只有检索器身份、语料、数据集 hash、TopK、查询身份、gold/forbidden 引用、运行时、干净 revision、smoke 状态和安全门禁全部一致，比较工具才接受三路报告。

这些报告仍属于 `smoke`：它们证明一条固定本地路径曾经运行成功，但合成程度过高，不能支持通用检索质量声明。不得根据已观察到的排名修改 v3 的查询文本或 gold；任何变更都必须创建新数据集版本。benchmark v1 同样遵守冻结规则。

LM Studio 会返回配置的模型名称，但不会提供可验证的权重 revision。因此，`profile_id` 是由操作者管理的显式向量空间边界，不能证明模型权重已被密码学固定。

指标定义：

- `hit_rate_at_k`：Top K 中至少出现一个 gold 的查询比例；
- `recall_at_k`：每条查询的 `已召回 gold / 全部 gold` 在 K 上的均值；
- `mrr`：第一个 gold 的倒数排名；
- `ndcg_at_k`：K 上的二元相关性排序质量。

## 验证

```bash
go test ./...
go vet ./...
RAGHUB_TEST_DATABASE_URL="$RAGHUB_DATABASE_URL" \
  go test -count=1 ./internal/store/postgres
```

也可以使用 Makefile：

```bash
make db-up
make verify
make test-integration
make eval-all
```

面向发布的门禁还会运行 race detector、校验 module checksum、检查格式，并扫描 Go 可达代码中的已知漏洞：

```bash
GOSUMDB=sum.golang.org make verify-release
```

未设置 `RAGHUB_TEST_DATABASE_URL` 时，集成测试会跳过；被跳过的测试不属于 PostgreSQL 运行证据。

## API 与设计

- [OpenAPI 契约](api/openapi.yaml)
- [ADR 0001：可度量的 PostgreSQL FTS 切片](docs/adr/0001-first-retrieval-slice.md)
- [ADR 0002：精确 pgvector Dense 基线](docs/adr/0002-exact-pgvector-dense-baseline.md)
- [ADR 0003：失败关闭的 Hybrid RRF 基线](docs/adr/0003-hybrid-rrf-baseline.md)
- [ADR 0004：困难 benchmark 与精确候选诊断](docs/adr/0004-hard-benchmark-candidate-diagnosis.md)
- [历史 benchmark v1 实验](docs/experiments/2026-08-20-benchmark-v1.md)
- [当前发布候选 benchmark](docs/experiments/2026-09-02-benchmark-v1-release-candidate.md)
- [`v0.1.0-alpha` 发布验收记录](docs/releases/v0.1.0-alpha-readiness.md)
- [数据库迁移](migrations/)

## 项目规范

- [Apache-2.0 许可证](LICENSE)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [行为准则](CODE_OF_CONDUCT.md)
- [变更日志](CHANGELOG.md)

公开源码不会改变当前运行边界。在处理真实数据或暴露到互联网之前，必须补充可信身份、速率和资源限制、密钥管理、传输安全、可观测性、备份恢复流程、数据保留控制，以及符合真实规模的负载和质量验证。仓库中的 benchmark 是合成数据，其结果不能证明通用检索质量或生产性能。

## 后续阶段

1. 启用仓库安全控制并发布已经验收的 alpha release。
2. 独立收集长文档、版本切换、噪声查询和经过脱敏的真实 bad case，创建新版本数据集，不修改冻结的 benchmark v1。
3. 只有预注册门禁观察到足够多且可恢复的 Hybrid 排序失败时，才引入 reranker。
4. 只有语料规模确有需要时，才评测租户感知 ANN，并以精确 Dense 基线校验其召回损失。
5. 在生产 pilot 前补齐可信身份边界、资源治理、OpenTelemetry trace、备份恢复证据和真实负载测试。

答案生成和 Agentic RAG 必须排在检索证据之后，而不是之前。
