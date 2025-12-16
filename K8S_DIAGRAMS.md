# OpenRSVP Kubernetes Architecture - Mermaid Diagrams

This document contains all the architectural diagrams in Mermaid.js format for the OpenRSVP Kubernetes scale-out project.

---

## Table of Contents
1. [Current Architecture](#1-current-architecture-monolithic)
2. [Target Architecture (AWS)](#2-target-architecture-aws--kubernetes)
3. [Detailed Component Flow](#3-detailed-component-flow)
4. [CI/CD Deployment Pipeline](#4-cicd-deployment-pipeline)
5. [Data Migration Flow](#5-data-migration-flow)
6. [Horizontal Pod Autoscaling](#6-horizontal-pod-autoscaling-hpa)
7. [Observability Stack](#7-observability-stack)
8. [Network Architecture](#8-network-architecture-aws)
9. [Worker Job Execution](#9-worker-job-execution-patterns)

---

## 1. Current Architecture (Monolithic)

**Description**: Current single-container deployment with SQLite

```mermaid
graph TB
    subgraph Container["Monolithic Container"]
        Web["FastAPI Web Server<br/>(handles all HTTP)"]
        Scheduler["APScheduler<br/>- decay cycle (hourly)<br/>- vacuum (every 12hrs)"]
        DB["SQLite Database<br/>(data/openrsvp.db)"]
    end
    
    style Container fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Web fill:#e1f5ff,stroke:#01579b
    style Scheduler fill:#fff3e0,stroke:#e65100
    style DB fill:#f3e5f5,stroke:#4a148c
```

---

## 2. Target Architecture (AWS + Kubernetes)

**Description**: Multi-tier architecture with EKS, web pods, worker pods, and RDS PostgreSQL

```mermaid
graph TB
    Internet([Internet]) --> Route53[Route53 DNS]
    Route53 --> ALB[ACM + ALB<br/>SSL/TLS]
    
    ALB --> EKS[EKS Cluster AWS]
    
    subgraph EKS["EKS Cluster (AWS)"]
        subgraph WebTier["Web Tier (Deployment)"]
            Web1[Pod 1: FastAPI<br/>scheduler disabled]
            Web2[Pod 2: FastAPI<br/>scheduler disabled]
            WebN[Pod N: FastAPI HPA<br/>scheduler disabled]
        end
        
        subgraph WorkerTier["Worker Tier (CronJob/Deploy)"]
            CronDecay[CronJob: decay-cycle]
            CronVacuum[CronJob: vacuum-db]
        end
    end
    
    WebTier --> RDS[(RDS PostgreSQL<br/>Multi-AZ<br/>Automated backups)]
    WorkerTier --> RDS
    
    style Internet fill:#fff,stroke:#333
    style Route53 fill:#ff9900,stroke:#333
    style ALB fill:#ff9900,stroke:#333
    style EKS fill:#f0f0f0,stroke:#333,stroke-width:3px
    style WebTier fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style WorkerTier fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style RDS fill:#527fff,stroke:#333
```

---

## 3. Detailed Component Flow

**Description**: Internal Kubernetes service mesh and database connections

```mermaid
graph LR
    User([User]) --> Ingress[Ingress Controller<br/>nginx/Traefik<br/>TLS + Rate Limiting]
    
    Ingress --> WebSvc[Service: ClusterIP<br/>openrsvp-web]
    
    WebSvc --> Web1[Web Pod 1]
    WebSvc --> Web2[Web Pod 2]
    WebSvc --> Web3[Web Pod 3+<br/>HPA Enabled]
    
    Web1 --> PG[(PostgreSQL<br/>Service)]
    Web2 --> PG
    Web3 --> PG
    
    Worker[Worker Pod<br/>Replicas: 1<br/>Scheduler Enabled] --> PG
    
    subgraph "Database Layer"
        PG --> PGDB[(PostgreSQL<br/>StatefulSet or<br/>Managed Service)]
    end
    
    style User fill:#fff,stroke:#333
    style Ingress fill:#4caf50,stroke:#333
    style WebSvc fill:#2196f3,stroke:#333
    style Web1 fill:#e1f5ff,stroke:#01579b
    style Web2 fill:#e1f5ff,stroke:#01579b
    style Web3 fill:#e1f5ff,stroke:#01579b
    style Worker fill:#fff3e0,stroke:#e65100
    style PG fill:#527fff,stroke:#333
    style PGDB fill:#527fff,stroke:#333
```

---

## 4. CI/CD Deployment Pipeline

**Description**: Automated build, test, and deployment flow

```mermaid
flowchart LR
    Dev[Developer] -->|git push| Repo[GitHub Repo]
    Repo -->|webhook| CI[CI/CD Pipeline<br/>GitHub Actions]
    
    CI --> Build[Build Docker Image]
    Build --> Test[Run Tests<br/>pytest]
    Test --> Push[Push to Registry<br/>ECR/DockerHub]
    
    Push --> Deploy{Deploy Stage}
    Deploy -->|dev| DevK8s[Dev K8s Cluster<br/>Bare Metal]
    Deploy -->|prod| ProdK8s[Prod EKS Cluster<br/>AWS]
    
    ProdK8s --> Rolling[Rolling Update<br/>Web Pods]
    ProdK8s --> CronUpdate[Update CronJobs<br/>Workers]
    
    style Dev fill:#fff,stroke:#333
    style Repo fill:#24292e,stroke:#333
    style CI fill:#2088ff,stroke:#333
    style Build fill:#ffa500,stroke:#333
    style Test fill:#4caf50,stroke:#333
    style Push fill:#00bcd4,stroke:#333
    style Deploy fill:#ff9800,stroke:#333
    style DevK8s fill:#e1f5ff,stroke:#333
    style ProdK8s fill:#ff9900,stroke:#333
```

---

## 5. Data Migration Flow

**Description**: SQLite to PostgreSQL migration process

```mermaid
flowchart TD
    Start([Start Migration]) --> Backup[Backup SQLite DB]
    Backup --> Export[Export Data<br/>Events, RSVPs,<br/>Channels, Messages]
    
    Export --> SetupPG[Setup PostgreSQL<br/>RDS or K8s]
    SetupPG --> RunMigrations[Run Alembic Migrations]
    
    RunMigrations --> Import[Import Data<br/>to PostgreSQL]
    Import --> Validate[Validate Data Integrity<br/>Record Counts, Tokens]
    
    Validate --> Test{Tests Pass?}
    Test -->|No| Debug[Debug Issues]
    Debug --> Import
    
    Test -->|Yes| Switch[Update Config<br/>Point to PostgreSQL]
    Switch --> Deploy[Deploy New Version]
    Deploy --> Monitor[Monitor for Issues]
    
    Monitor --> Done([Migration Complete])
    
    style Start fill:#4caf50,stroke:#333
    style Backup fill:#ff9800,stroke:#333
    style Export fill:#2196f3,stroke:#333
    style SetupPG fill:#9c27b0,stroke:#333
    style Import fill:#2196f3,stroke:#333
    style Validate fill:#ffc107,stroke:#333
    style Test fill:#ff9800,stroke:#333
    style Debug fill:#f44336,stroke:#333
    style Switch fill:#4caf50,stroke:#333
    style Deploy fill:#4caf50,stroke:#333
    style Done fill:#4caf50,stroke:#333
```

---

## 6. Horizontal Pod Autoscaling (HPA)

**Description**: How web tier scales based on CPU/memory metrics

```mermaid
graph TB
    Metrics[Metrics Server] -->|CPU/Memory| HPA[Horizontal Pod<br/>Autoscaler]
    
    HPA -->|Scale Up/Down| Deployment[Web Deployment]
    
    Deployment --> Pod1[Pod 1]
    Deployment --> Pod2[Pod 2]
    Deployment -.->|Scale to| PodN[Pod N<br/>Max: 10]
    
    LoadBalancer[Load Balancer] --> Pod1
    LoadBalancer --> Pod2
    LoadBalancer --> PodN
    
    Config[Config:<br/>Min: 2<br/>Max: 10<br/>Target CPU: 70%]
    
    style Metrics fill:#4caf50,stroke:#333
    style HPA fill:#ff9800,stroke:#333
    style Deployment fill:#2196f3,stroke:#333
    style Pod1 fill:#e1f5ff,stroke:#333
    style Pod2 fill:#e1f5ff,stroke:#333
    style PodN fill:#e1f5ff,stroke:#333,stroke-dasharray: 5 5
    style LoadBalancer fill:#9c27b0,stroke:#333
    style Config fill:#ffc107,stroke:#333
```

---

## 7. Observability Stack

**Description**: Monitoring, logging, and alerting infrastructure

```mermaid
graph TB
    subgraph Apps["Application Layer"]
        WebPods[Web Pods]
        WorkerPods[Worker Pods]
    end
    
    subgraph Metrics["Metrics Collection"]
        WebPods -->|metrics| Prometheus[Prometheus]
        WorkerPods -->|metrics| Prometheus
        Prometheus --> Grafana[Grafana Dashboards]
    end
    
    subgraph Logs["Log Aggregation"]
        WebPods -->|logs| Loki[Loki / CloudWatch]
        WorkerPods -->|logs| Loki
        Loki --> GrafanaLogs[Grafana Log Viewer]
    end
    
    subgraph Alerts["Alerting"]
        Prometheus --> AlertManager[Alert Manager]
        AlertManager --> Slack[Slack]
        AlertManager --> Email[Email]
        AlertManager --> PagerDuty[PagerDuty]
    end
    
    subgraph Tracing["Distributed Tracing (Optional)"]
        WebPods -.->|traces| Jaeger[Jaeger / Tempo]
        Jaeger -.-> GrafanaTraces[Grafana Traces]
    end
    
    style WebPods fill:#e1f5ff,stroke:#333
    style WorkerPods fill:#fff3e0,stroke:#333
    style Prometheus fill:#e6522c,stroke:#333
    style Grafana fill:#f46800,stroke:#333
    style Loki fill:#00a6fb,stroke:#333
    style AlertManager fill:#ff6b6b,stroke:#333
    style Jaeger fill:#60d0e4,stroke:#333
```

---

## 8. Network Architecture (AWS)

**Description**: VPC layout with public/private subnets and security groups

```mermaid
graph TB
    Internet([Internet]) --> IGW[Internet Gateway]
    
    subgraph VPC["VPC (10.0.0.0/16)"]
        subgraph PublicSubnet["Public Subnet (10.0.1.0/24)"]
            ALB[Application Load<br/>Balancer]
            NAT[NAT Gateway]
        end
        
        subgraph PrivateSubnetA["Private Subnet A (10.0.10.0/24)"]
            EKSNodeA[EKS Worker Nodes]
        end
        
        subgraph PrivateSubnetB["Private Subnet B (10.0.11.0/24)"]
            EKSNodeB[EKS Worker Nodes]
        end
        
        subgraph DataSubnetA["Data Subnet A (10.0.20.0/24)"]
            RDSA[(RDS Primary)]
        end
        
        subgraph DataSubnetB["Data Subnet B (10.0.21.0/24)"]
            RDSB[(RDS Standby<br/>Multi-AZ)]
        end
    end
    
    IGW --> ALB
    ALB --> EKSNodeA
    ALB --> EKSNodeB
    
    EKSNodeA --> NAT
    EKSNodeB --> NAT
    NAT --> IGW
    
    EKSNodeA --> RDSA
    EKSNodeB --> RDSA
    RDSA -.->|Replication| RDSB
    
    SG_ALB[Security Group: ALB<br/>Ingress: 80, 443]
    SG_EKS[Security Group: EKS<br/>Ingress: from ALB]
    SG_RDS[Security Group: RDS<br/>Ingress: 5432 from EKS]
    
    style Internet fill:#fff,stroke:#333
    style IGW fill:#ff9900,stroke:#333
    style VPC fill:#f0f0f0,stroke:#333,stroke-width:3px
    style PublicSubnet fill:#d4edda,stroke:#155724,stroke-width:2px
    style PrivateSubnetA fill:#d1ecf1,stroke:#0c5460,stroke-width:2px
    style PrivateSubnetB fill:#d1ecf1,stroke:#0c5460,stroke-width:2px
    style DataSubnetA fill:#f8d7da,stroke:#721c24,stroke-width:2px
    style DataSubnetB fill:#f8d7da,stroke:#721c24,stroke-width:2px
    style RDSA fill:#527fff,stroke:#333
    style RDSB fill:#527fff,stroke:#333,stroke-dasharray: 5 5
```

---

## 9. Worker Job Execution Patterns

**Description**: Three options for running background jobs in Kubernetes

### Option A: APScheduler in Deployment

```mermaid
graph LR
    Scheduler[APScheduler] --> DecayJob[Decay Cycle<br/>Every 1 hour]
    Scheduler --> VacuumJob[Vacuum DB<br/>Every 12 hours]
    
    subgraph WorkerPod["Worker Pod (Replicas: 1)"]
        Scheduler
        DecayJob
        VacuumJob
    end
    
    DecayJob --> DB[(PostgreSQL)]
    VacuumJob --> DB
    
    Note[Note: Single pod to avoid<br/>duplicate job execution]
    
    style Scheduler fill:#fff3e0,stroke:#333
    style DecayJob fill:#ffccbc,stroke:#333
    style VacuumJob fill:#ffccbc,stroke:#333
    style WorkerPod fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style DB fill:#527fff,stroke:#333
    style Note fill:#fff9c4,stroke:#333
```

### Option B: Kubernetes CronJobs (Recommended)

```mermaid
graph LR
    K8s[Kubernetes Scheduler]
    
    K8s -->|Every hour| CronDecay[CronJob: decay-cycle]
    K8s -->|Every 12 hours| CronVacuum[CronJob: vacuum-db]
    
    CronDecay --> PodDecay[Job Pod<br/>Runs decay script]
    CronVacuum --> PodVacuum[Job Pod<br/>Runs vacuum script]
    
    PodDecay --> DB[(PostgreSQL)]
    PodVacuum --> DB
    
    PodDecay -.->|Completes| Cleanup1[Pod Deleted]
    PodVacuum -.->|Completes| Cleanup2[Pod Deleted]
    
    style K8s fill:#4caf50,stroke:#333
    style CronDecay fill:#fff3e0,stroke:#333
    style CronVacuum fill:#fff3e0,stroke:#333
    style PodDecay fill:#ffccbc,stroke:#333
    style PodVacuum fill:#ffccbc,stroke:#333
    style DB fill:#527fff,stroke:#333
    style Cleanup1 fill:#e0e0e0,stroke:#333
    style Cleanup2 fill:#e0e0e0,stroke:#333
```

### Option C: Distributed Task Queue (Future)

```mermaid
graph LR
    API[FastAPI Web] -->|Enqueue| Redis[(Redis Queue)]
    Scheduler[APScheduler] -->|Enqueue| Redis
    
    Redis --> Worker1[Worker 1<br/>Arq/Celery]
    Redis --> Worker2[Worker 2<br/>Arq/Celery]
    Redis --> WorkerN[Worker N<br/>Arq/Celery]
    
    Worker1 --> DB[(PostgreSQL)]
    Worker2 --> DB
    WorkerN --> DB
    
    Worker1 --> Email[Email Service<br/>SES]
    Worker2 --> Email
    
    Note[Note: For future features<br/>like email notifications,<br/>webhooks, etc.]
    
    style API fill:#e1f5ff,stroke:#333
    style Scheduler fill:#fff3e0,stroke:#333
    style Redis fill:#dc382d,stroke:#333
    style Worker1 fill:#ffccbc,stroke:#333
    style Worker2 fill:#ffccbc,stroke:#333
    style WorkerN fill:#ffccbc,stroke:#333,stroke-dasharray: 5 5
    style DB fill:#527fff,stroke:#333
    style Email fill:#00bcd4,stroke:#333
    style Note fill:#fff9c4,stroke:#333
```

---

## Usage Notes

### Viewing Mermaid Diagrams

These diagrams will render automatically on:
- **GitHub** - in README.md or any .md file
- **GitLab** - in README.md or any .md file
- **VS Code** - with "Markdown Preview Mermaid Support" extension
- **Obsidian** - native support
- **Notion** - using mermaid code blocks

### Editing Diagrams

You can edit these diagrams using:
- **Mermaid Live Editor**: https://mermaid.live/
- **VS Code** with Mermaid extension
- Any text editor (it's just text!)

### Exporting Diagrams

To export as PNG/SVG:
1. Copy the mermaid code
2. Paste into https://mermaid.live/
3. Click "Download PNG" or "Download SVG"

---

## Diagram Legend

### Colors
- **Blue (#e1f5ff)**: Web tier / Application pods
- **Orange (#fff3e0)**: Worker tier / Background jobs
- **Purple (#527fff)**: Databases / Data storage
- **Green (#4caf50)**: Monitoring / Health systems
- **Yellow (#ff9900)**: AWS services
- **Red (#f44336)**: Errors / Alerts

### Line Styles
- **Solid line** (─): Active/implemented
- **Dashed line** (⋯): Optional/future
- **Arrow** (→): Data flow direction

---

## Related Documentation

- Main planning document: [K8S_SCALE_PLAN.md](./K8S_SCALE_PLAN.md)
- Current architecture: See `openrsvp/` directory
- Docker setup: See `docker-compose.yml` and `Dockerfile`

