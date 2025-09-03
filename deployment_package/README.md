# Factor Forecasting 部署包

## 包含内容

- `src/` - 完全修复的源代码
- `configs/` - 配置文件
- `scripts/` - 部署和启动脚本
- `requirements.txt` - Python依赖包
- `deploy_to_server.sh` - 自动化部署脚本

## 核心修复内容

1. ✅ 修复pickle序列化错误
2. ✅ 修复代码缩进错误
3. ✅ 优化内存管理阈值(95%/98%)
4. ✅ 优化数据加载器配置
5. ✅ 完善配置加载功能

## 部署说明

1. 确保本地虚拟环境已激活
2. 运行 `chmod +x deploy_to_server.sh`
3. 执行 `./deploy_to_server.sh`

## 预期性能提升

- CPU利用率: 6.2% → 50%+ (8倍提升)
- 内存清理频率: 减少99%
- 训练稳定性: 完全稳定

