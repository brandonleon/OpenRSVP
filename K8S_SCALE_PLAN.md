# OpenRSVP Kubernetes Scale-Out Architecture Plan

**Status**: Planning Phase  
**Created**: 2025-12-09  
**Target Go-Live**: TBD  

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Architecture](#current-architecture)
3. [Target Architecture](#target-architecture)
4. [Requirements Questionnaire](#requirements-questionnaire)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Cost Estimates](#cost-estimates)
7. [Risk Assessment](#risk-assessment)
8. [Success Metrics](#success-metrics)

---

## Executive Summary

OpenRSVP is currently a monolithic FastAPI application using SQLite. To prepare for production launch at scale, we need to:

1. **Migrate to PostgreSQL** - Enable multi-node deployment
2. **Separate web and worker tiers** - Independent scaling
3. **Deploy to Kubernetes** - Production on AWS EKS, Development on bare metal
4. **Implement observability** - Monitoring, logging, alerting
5. **Automate CI/CD** - Streamlined deployments

**Key Goals**:
- ✅ Scale-ready before launch (no rush, build it right)
- ✅ Support 10x-100x growth without architectural changes
- ✅ Minimize operational burden
- ✅ Cost-effective for early stage

---

## Current Architecture

### Application Stack
- **Framework**: FastAPI + Uvicorn
- **Database**: SQLite (single file, not distributed)
- **Scheduler**: APScheduler (in-process background jobs)
- **Deployment**: Docker + Docker Compose
- **Static Assets**: Served from container filesystem

### Current Components
```
┌─────────────────────────────────────┐
│     Monolithic Container            │
│  ┌──────────────────────────────┐  │
│  │   FastAPI Web Server         │  │
│  │   (handles all HTTP)         │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   APScheduler                │  │
│  │   - decay cycle (hourly)     │  │
│  │   - vacuum (every 12hrs)     │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   SQLite Database            │  │
│  │   (data/openrsvp.db)         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Background Jobs
1. **Decay Cycle** (`openrsvp/decay.py:run_decay_cycle`)
   - Runs every `decay_interval_hours` (default: 1 hour)
   - Applies exponential decay to event/channel scores
   - Deletes events below threshold after grace period
   - Batch processing (200 records at a time)

2. **Database Vacuum** (`openrsvp/decay.py:vacuum_database`)
   - Runs every `sqlite_vacuum_hours` (default: 12 hours)
   - SQLite-specific optimization (needs Postgres equivalent)

### Key Files
- `openrsvp/api.py` (2000+ lines) - Main FastAPI app
- `openrsvp/scheduler.py` - APScheduler integration
- `openrsvp/decay.py` - Background job logic
- `openrsvp/database.py` - SQLAlchemy engine setup
- `openrsvp/config.py` - Configuration management
- `docker-compose.yml` - Current deployment

---

## Target Architecture

### High-Level Design

```
                    ┌─────────────────┐
                    │   Route53 DNS   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   ACM + ALB     │
                    │  (SSL/TLS)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │         EKS Cluster (AWS)               │
        │  ┌──────────────────────────────────┐  │
        │  │  Web Tier (Deployment)           │  │
        │  │  ├─ Pod 1: FastAPI               │  │
        │  │  ├─ Pod 2: FastAPI               │  │
        │  │  └─ Pod N: FastAPI (HPA)         │  │
        │  │     - scheduler disabled         │  │
        │  └──────────────────────────────────┘  │
        │  ┌──────────────────────────────────┐  │
        │  │  Worker Tier (CronJob/Deploy)    │  │
        │  │  ├─ CronJob: decay-cycle         │  │
        │  │  └─ CronJob: vacuum-db           │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬─────────────────────┘
                         │
                ┌────────▼────────┐
                │  RDS PostgreSQL │
                │  (Multi-AZ?)    │
                │  - Automated    │
                │    backups      │
                └─────────────────┘
```

### Component Breakdown

#### Web Tier
- **Purpose**: Handle HTTP traffic (RSVP creation, event management, admin UI)
- **Scaling**: Horizontal Pod Autoscaler (HPA) based on CPU/memory
- **Configuration**: `enable_scheduler = False`
- **Replicas**: 
  - Minimum: 2 (high availability)
  - Maximum: 10+ (scale with traffic)
  - Target CPU: 70%

#### Worker Tier
- **Purpose**: Background jobs (decay, vacuum, future: emails)
- **Scaling**: Static replica count or CronJobs
- **Configuration**: `enable_scheduler = True` (if using deployment) or K8s CronJobs
- **Options**:
  - **Option A**: Single Deployment pod with APScheduler
  - **Option B**: Kubernetes CronJobs (more K8s-native)
  - **Option C**: Distributed task queue (Arq/Celery) - future

#### Database
- **PostgreSQL** (replacing SQLite)
- **Why**: Multi-node access, better concurrency, production-grade
- **Hosting Options**:
  - AWS RDS (managed) - **RECOMMENDED**
  - Self-hosted in K8s StatefulSet
- **Migration**: Export SQLite → Import PostgreSQL (downtime acceptable)

---

## Requirements Questionnaire

**Instructions**: Please fill out the sections below. Add notes, concerns, or questions anywhere. We'll review together before implementation.

---

### 1. AWS Infrastructure

#### 1.1 Kubernetes Platform
**Question**: For AWS, which Kubernetes platform?

- [ ] **EKS** (Elastic Kubernetes Service) - Managed control plane, you manage worker nodes
  - Pros: AWS-native, good integration, mature
  - Cons: $73/mo for control plane, more complex than ECS
  
- [ ] **ECS with Fargate** - Serverless containers (not Kubernetes)
  - Pros: Simpler, no control plane cost, pay-per-use
  - Cons: Less portable, vendor lock-in, missing K8s features
  
- [ ] **EKS with Fargate** - Serverless K8s pods
  - Pros: No node management, K8s API
  - Cons: Higher pod costs, some K8s features limited

**Your Choice**: ___________________________

**Notes**: 


#### 1.2 Database Strategy
**Question**: Where should the production database run?

- [ ] **RDS PostgreSQL** (fully managed)
  - Pros: Automated backups, patching, Multi-AZ HA, monitoring
  - Cons: ~$30-120/mo depending on size, slight vendor lock-in
  
- [ ] **PostgreSQL in K8s** (StatefulSet with operator)
  - Pros: Full control, no RDS cost, portable
  - Cons: You manage backups, HA, patching, storage

**Your Choice**: ___________________________

**Database Size** (if RDS):
- [ ] db.t4g.micro - $12/mo (dev/staging only)
- [ ] db.t4g.small - $29/mo (small prod)
- [ ] db.t4g.medium - $58/mo (medium prod)
- [ ] db.t4g.large - $116/mo (large prod)

**High Availability**:
- [ ] Single-AZ (cheaper, ~5 min downtime during maintenance)
- [ ] Multi-AZ (2x cost, 99.95% SLA, automatic failover)

**Your Choices**: 
- Size: ___________________________
- HA: ___________________________

**Notes**:


#### 1.3 Load Balancer
**Question**: Which AWS load balancer?

- [ ] **ALB** (Application Load Balancer) - HTTP/HTTPS, path-based routing
  - Cost: ~$16/mo + data transfer
  - Best for: Web applications (recommended)
  
- [ ] **NLB** (Network Load Balancer) - TCP/UDP, ultra-low latency
  - Cost: ~$16/mo + data transfer
  - Best for: Non-HTTP protocols or extreme performance needs

**Your Choice**: ___________________________

**Notes**:


#### 1.4 AWS Region
**Question**: Which AWS region for production?

Primary region: ___________________________  
(e.g., us-east-1, us-west-2, eu-west-1)

**Multi-region needed?**: [ ] Yes [ ] No

**Notes**:


#### 1.5 Existing AWS Infrastructure
Do you have:
- [ ] Existing AWS account
- [ ] Existing VPC setup
- [ ] AWS credits/budget approval
- [ ] IAM users/roles configured
- [ ] Other AWS services in use (specify): ___________________________

**Notes**:


---

### 2. Development Environment

#### 2.1 Bare Metal K8s Setup
**Question**: What K8s distribution for your bare metal dev cluster?

- [ ] **k3s** (lightweight, easy to install)
- [ ] **microk8s** (Canonical/Ubuntu-focused)
- [ ] **minikube** (local dev, single node)
- [ ] **kind** (Kubernetes in Docker)
- [ ] **kubeadm** (vanilla K8s)
- [ ] **Rancher** (full platform with UI)
- [ ] Other: ___________________________

**Your Choice**: ___________________________

**Current Status**:
- [ ] Already installed and running
- [ ] Need setup instructions
- [ ] Not sure yet

**Cluster Size**:
- Number of nodes: ___________________________
- Node specs (CPU/RAM): ___________________________

**Notes**:


#### 2.2 Dev Database
**Question**: How should database run in dev environment?

- [ ] PostgreSQL in K8s (StatefulSet)
- [ ] PostgreSQL in Docker (separate from K8s)
- [ ] Shared dev database (cloud-hosted)
- [ ] Each developer runs own Postgres

**Your Choice**: ___________________________

**Notes**:


---

### 3. Scale & Performance Targets

#### 3.1 Expected Traffic
**Launch scale** (first 3 months):
- Events per month: ___________________________
- RSVPs per event (average): ___________________________
- RSVPs per event (peak): ___________________________
- Concurrent users (peak): ___________________________

**Growth expectations** (next 12 months):
- [ ] Steady (2x growth)
- [ ] Moderate (5x growth)
- [ ] Aggressive (10x+ growth)
- [ ] Viral potential (unpredictable spikes)

**Geographic distribution**:
- [ ] Single region (specify): ___________________________
- [ ] Multi-region (specify): ___________________________
- [ ] Global

**Notes**:


#### 3.2 Performance Requirements
**Latency targets**:
- Page load (p95): ___________________________ms (e.g., < 500ms)
- API response (p95): ___________________________ms (e.g., < 200ms)

**Uptime target**:
- [ ] 99% (7.2 hours downtime/month)
- [ ] 99.9% (43 minutes downtime/month)
- [ ] 99.95% (21 minutes downtime/month)
- [ ] 99.99% (4 minutes downtime/month)

**Notes**:


---

### 4. Budget & Cost

#### 4.1 Monthly Infrastructure Budget
What's your comfortable monthly AWS spend?

- [ ] **Minimal** ($50-100/mo)
  - Small RDS single-AZ, 2 small nodes, basic monitoring
  
- [ ] **Standard** ($200-400/mo)
  - Medium RDS Multi-AZ, 3 medium nodes, full monitoring
  
- [ ] **Production-grade** ($500-1000/mo)
  - Large RDS Multi-AZ + replicas, autoscaling nodes, premium support

**Your Budget**: $___________________________/mo

**Cost priorities**:
- [ ] Minimize cost (willing to accept some downtime/slower perf)
- [ ] Balance cost and reliability
- [ ] Optimize for reliability (cost secondary)

**Notes**:


---

### 5. Architecture Decisions

#### 5.1 Background Job Strategy
**Question**: How should scheduled jobs (decay, vacuum) run?

- [ ] **Option A**: APScheduler in single worker pod
  - Pros: Works now, no code changes
  - Cons: Single point of failure, can't scale horizontally
  
- [ ] **Option B**: Kubernetes CronJobs
  - Pros: K8s-native, reliable, easy to scale
  - Cons: Need to refactor scheduler code
  - **RECOMMENDED** for K8s deployment
  
- [ ] **Option C**: Distributed task queue (Arq/Celery + Redis)
  - Pros: Production-grade, handles complex workflows
  - Cons: More infrastructure, overkill for current needs

**Your Choice**: ___________________________

**Notes**:


#### 5.2 Caching Layer (Redis)
**Question**: Add Redis caching from day 1?

**Use cases**:
- Rate limiting
- Session storage (if you add auth)
- Query result caching
- Job queue backing (if using Arq/Celery)

- [ ] **Yes, add Redis now** (~$15-30/mo for ElastiCache or in-cluster)
- [ ] **No, add later when metrics show need** (start simple)

**Your Choice**: ___________________________

**If Yes, where?**:
- [ ] AWS ElastiCache (managed)
- [ ] Redis in K8s (self-hosted)

**Notes**:


#### 5.3 Static Assets Strategy
**Question**: How to serve CSS/JS/images?

- [ ] **Option A**: Serve from app pods (current)
  - Pros: Simple, no extra cost
  - Cons: Uses app resources, no global distribution
  
- [ ] **Option B**: S3 + CloudFront CDN
  - Pros: Offload traffic, global performance, cheap
  - Cons: Deployment complexity, cache invalidation
  - Cost: ~$5-20/mo depending on traffic

**Your Choice**: ___________________________

**User uploads planned?** (event images, attachments)
- [ ] Yes - will need S3
- [ ] No - can skip for now
- [ ] Maybe later

**Notes**:


---

### 6. Development & Deployment

#### 6.1 CI/CD Platform
**Question**: Which CI/CD system?

- [ ] **GitHub Actions** (easiest if using GitHub)
- [ ] **GitLab CI** (if using GitLab)
- [ ] **AWS CodePipeline** (AWS-native)
- [ ] **Jenkins** (self-hosted)
- [ ] Other: ___________________________

**Your Choice**: ___________________________

**Current repo location**: ___________________________

**Notes**:


#### 6.2 Deployment Strategy
**Question**: How should deployments roll out?

- [ ] **Rolling update** (gradual pod replacement)
  - Pros: Simple, no extra resources
  - Cons: Mixed versions during deploy
  
- [ ] **Blue/Green** (full environment swap)
  - Pros: Zero downtime, instant rollback
  - Cons: 2x resources during deploy
  
- [ ] **Canary** (test on small % of traffic first)
  - Pros: Catch issues early
  - Cons: More complex

**Your Choice**: ___________________________

**Notes**:


#### 6.3 Infrastructure as Code
**Question**: How to manage AWS infrastructure?

- [ ] **Terraform** (most popular, cloud-agnostic)
- [ ] **CloudFormation** (AWS-native)
- [ ] **Pulumi** (code-based)
- [ ] **CDK** (AWS Cloud Development Kit)
- [ ] Manual (ClickOps) - not recommended

**Your Choice**: ___________________________

**K8s manifests format**:
- [ ] **Helm charts** (templated, reusable)
- [ ] **Raw YAML** (simpler to understand)
- [ ] **Kustomize** (overlays)

**Your Choice**: ___________________________

**Notes**:


#### 6.4 Secrets Management
**Question**: How to manage secrets (DB passwords, API keys)?

- [ ] **AWS Secrets Manager** ($0.40/secret/mo)
- [ ] **AWS SSM Parameter Store** (free for standard params)
- [ ] **External Secrets Operator** (sync AWS → K8s)
- [ ] **Sealed Secrets** (encrypted in Git)
- [ ] **HashiCorp Vault** (enterprise-grade)

**Your Choice**: ___________________________

**Notes**:


---

### 7. Observability

#### 7.1 Monitoring Stack
**Question**: What level of monitoring do you need?

- [ ] **Minimal** - CloudWatch metrics + health checks
  - Cost: Included with AWS
  
- [ ] **Standard** - Prometheus + Grafana in-cluster
  - Cost: ~$20-50/mo in cluster resources
  
- [ ] **Full** - Above + ELK/Loki for logs + tracing
  - Cost: ~$100-200/mo

**Your Choice**: ___________________________

**Comfort level operating monitoring infrastructure**:
- [ ] Comfortable (I've run Prometheus/Grafana before)
- [ ] Learning (willing to learn)
- [ ] Prefer managed (use AWS CloudWatch/Managed Prometheus)

**Notes**:


#### 7.2 Logging Strategy
**Question**: Where should logs be stored?

- [ ] **CloudWatch Logs** (AWS-native, pay per GB)
- [ ] **ELK Stack** (Elasticsearch + Logstash + Kibana)
- [ ] **Loki** (like Prometheus but for logs)
- [ ] **S3** (cheap archival, harder to query)

**Your Choice**: ___________________________

**Log retention period**: ___________________________days

**Notes**:


#### 7.3 Alerting
**Question**: How should you be notified of issues?

- [ ] **Email** (free, simple)
- [ ] **Slack** (webhook integration)
- [ ] **Discord** (webhook integration)
- [ ] **PagerDuty** (serious on-call, $$$)
- [ ] **Opsgenie** (alternative to PagerDuty)

**Your Choice**: ___________________________

**On-call tolerance**:
- [ ] Solo founder - need simple alerting
- [ ] Small team - can rotate on-call
- [ ] Not ready for 24/7 monitoring yet

**Notes**:


---

### 8. Security

#### 8.1 SSL/TLS Certificates
**Question**: How to handle HTTPS certificates?

- [ ] **AWS Certificate Manager (ACM)** - free, auto-renewal
  - Best for: ALB/CloudFront termination
  
- [ ] **Let's Encrypt + cert-manager** - free, K8s-native
  - Best for: In-cluster termination, dev environments

**Your Choice for Production**: ___________________________

**Your Choice for Development**: ___________________________

**Notes**:


#### 8.2 Network Security
**Configurations to consider**:

- [ ] RDS in private subnet (yes, recommended)
- [ ] EKS pods in private subnets with NAT Gateway (~$32/mo)
- [ ] Network policies between K8s pods
- [ ] Web Application Firewall (WAF) on ALB (~$5/mo + rules)

**Your Choices**:
- Private subnets: [ ] Yes [ ] No
- NAT Gateway: [ ] Yes [ ] No (save cost)
- Network policies: [ ] Now [ ] Later
- WAF: [ ] Now [ ] Later

**Notes**:


#### 8.3 IAM & Access Control
**Question**: Who needs access to what?

**AWS IAM**:
- Number of developers needing AWS access: ___________________________
- [ ] Use IAM users
- [ ] Use IAM roles + SSO
- [ ] Use AWS Organizations

**K8s RBAC**:
- [ ] Admin access for all devs
- [ ] Namespace-level permissions
- [ ] Read-only access for some users

**Notes**:


---

### 9. Future Features

**Question**: Which features are on your 6-12 month roadmap?

Check all that apply (helps us future-proof architecture):

- [ ] **Email notifications** (RSVP confirmations, reminders)
  - Needs: Email service (SES), task queue
  
- [ ] **Calendar integrations** (iCal export, Google Calendar)
  - Needs: Already supported via `openrsvp/utils/ics.py`
  
- [ ] **Payment processing** (paid events)
  - Needs: Stripe/PayPal integration, webhook handling
  
- [ ] **Multi-tenancy** (organizations/teams)
  - Needs: Schema changes, tenant isolation
  
- [ ] **Mobile app** (iOS/Android)
  - Needs: API versioning, push notifications
  
- [ ] **Webhooks** (integrate with Slack, Discord, etc.)
  - Needs: Task queue, retry logic
  
- [ ] **Analytics dashboard** (event insights, attendance trends)
  - Needs: Possibly data warehouse, BI tool
  
- [ ] **File uploads** (event images, attachments)
  - Needs: S3 storage, virus scanning
  
- [ ] **Search** (full-text event/channel search)
  - Needs: Possibly Elasticsearch/Typesense

**Your Priorities** (rank top 3):
1. ___________________________
2. ___________________________
3. ___________________________

**Notes**:


---

### 10. Migration & Data

#### 10.1 Current Beta Data
**Question**: Tell me about your existing beta site data

- Approximate number of events: ___________________________
- Approximate number of RSVPs: ___________________________
- Approximate number of channels: ___________________________
- Database size: ___________________________MB

**Data migration requirements**:
- [ ] Must preserve all data
- [ ] Can clean up test data
- [ ] Can start fresh (export/import only critical data)

**Downtime tolerance**:
- [ ] Can take site offline for migration (hours/days okay)
- [ ] Need to minimize downtime (< 1 hour)
- [ ] Need zero-downtime migration

**Notes**:


#### 10.2 Schema Stability
**Question**: Is your database schema still changing frequently?

- [ ] Stable - ready to freeze for production
- [ ] Minor changes expected
- [ ] Major changes coming - need flexibility

**Notes**:


---

### 11. Team & Operations

#### 11.1 Team Size
- Number of developers: ___________________________
- DevOps/Infrastructure experience level:
  - [ ] Beginner (need lots of documentation)
  - [ ] Intermediate (comfortable with basics)
  - [ ] Advanced (can troubleshoot complex issues)

**Notes**:


#### 11.2 Operational Preferences
**Question**: How much operational complexity are you willing to manage?

- [ ] **Minimal** - Maximize managed services, minimize ops burden
- [ ] **Balanced** - Mix of managed and self-hosted
- [ ] **Full control** - Self-host everything for learning/cost

**Your Preference**: ___________________________

**Time available for ops/maintenance**: ___________hours/week

**Notes**:


---

## Implementation Roadmap

*(This section will be filled in after you complete the questionnaire)*

### Phase 1: Database Migration
**Duration**: TBD  
**Effort**: TBD

- [ ] Add PostgreSQL support to codebase
- [ ] Update Alembic migrations
- [ ] Create migration scripts
- [ ] Test against Postgres locally
- [ ] Document rollback procedure

### Phase 2: Containerization
**Duration**: TBD  
**Effort**: TBD

- [ ] Multi-stage Dockerfile optimization
- [ ] Health check endpoints
- [ ] Graceful shutdown handling
- [ ] Separate entrypoints (web vs worker)

### Phase 3: Kubernetes Manifests
**Duration**: TBD  
**Effort**: TBD

- [ ] Web deployment + HPA
- [ ] Worker deployment/CronJobs
- [ ] Services (ClusterIP, LoadBalancer)
- [ ] Ingress configuration
- [ ] ConfigMaps and Secrets
- [ ] PersistentVolumeClaims (if needed)

### Phase 4: AWS Infrastructure
**Duration**: TBD  
**Effort**: TBD

- [ ] VPC setup (Terraform/CloudFormation)
- [ ] EKS cluster provisioning
- [ ] RDS PostgreSQL setup
- [ ] ALB configuration
- [ ] Route53 DNS
- [ ] ACM certificates

### Phase 5: CI/CD Pipeline
**Duration**: TBD  
**Effort**: TBD

- [ ] Build pipeline (Docker images)
- [ ] Test pipeline (pytest)
- [ ] Deploy pipeline (kubectl/Helm)
- [ ] Rollback procedures

### Phase 6: Observability
**Duration**: TBD  
**Effort**: TBD

- [ ] Prometheus + Grafana setup
- [ ] Application metrics
- [ ] Log aggregation
- [ ] Alerting rules
- [ ] Dashboards

### Phase 7: Production Launch
**Duration**: TBD  
**Effort**: TBD

- [ ] Data migration from beta
- [ ] DNS cutover
- [ ] Smoke tests
- [ ] Performance validation
- [ ] Runbook documentation

---

## Cost Estimates

*(Will be calculated based on your choices)*

### AWS Monthly Costs (Estimated)

| Component | Minimal | Standard | Production |
|-----------|---------|----------|------------|
| EKS Control Plane | $73 | $73 | $73 |
| EC2 Worker Nodes | $30 | $100 | $200+ |
| RDS PostgreSQL | $12-29 | $58-120 | $120-240 |
| Load Balancer | $16 | $16 | $16 |
| NAT Gateway | $0 | $32 | $64 |
| Data Transfer | ~$10 | ~$20 | ~$50 |
| CloudWatch/Logs | ~$5 | ~$20 | ~$50 |
| Backups | Included | ~$10 | ~$30 |
| **TOTAL** | **~$150** | **~$330** | **~$650+** |

**Notes**: 
- Costs vary by region and usage
- Does not include data transfer out to internet
- Does not include optional services (WAF, Shield, etc.)

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Database migration issues | High | Medium | Thorough testing, dry runs, rollback plan |
| SQLite → Postgres incompatibilities | Medium | Low | SQLAlchemy abstracts most differences |
| K8s complexity learning curve | Medium | Medium | Good documentation, start with simple setup |
| Cost overruns | Medium | Medium | Set AWS budgets, monitoring, right-sizing |
| Downtime during migration | Low | Low | Acceptable for pre-launch |

### Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Insufficient monitoring | High | Medium | Implement observability from day 1 |
| Security misconfiguration | High | Low | Security checklist, AWS Config rules |
| Scaling issues under load | Medium | Medium | Load testing before launch |
| Single point of failure | Medium | Low | Multi-AZ deployments, redundancy |

---

## Success Metrics

**Technical Metrics**:
- [ ] Web tier can scale 0 → 10 pods in < 5 minutes
- [ ] Database connection pooling prevents exhaustion
- [ ] p95 latency < 500ms for page loads
- [ ] Uptime > 99.9% (or your target)
- [ ] Zero-downtime deployments

**Operational Metrics**:
- [ ] Deploy time < 10 minutes
- [ ] MTTR (Mean Time To Recovery) < 30 minutes
- [ ] Can rollback deployment in < 5 minutes
- [ ] Alerts fire before users notice issues
- [ ] Cost within budget constraints

**Business Metrics**:
- [ ] Infrastructure costs < X% of revenue (once launched)
- [ ] Can support 10x growth without re-architecture
- [ ] Developer velocity maintained (fast iteration)

---

## Next Steps

1. **Fill out this questionnaire** - Answer all the questions above
2. **Review together** - We'll discuss your answers and clarify any questions
3. **Finalize architecture** - Lock in design decisions
4. **Create detailed implementation plan** - Break down into executable tasks
5. **Begin implementation** - Start with Phase 1 (database migration)

---

## Questions / Notes

*(Use this space for any questions or concerns)*



