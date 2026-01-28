# Voice-to-FHIR Reference Architecture

Deployment patterns for clinical voice documentation pipelines using MedASR and MedGemma.

---

## Table of Contents

1. [Overview](#overview)
2. [Component Architecture](#component-architecture)
3. [Deployment Pattern A: Small Practice](#deployment-pattern-a-small-practice)
4. [Deployment Pattern B: Enterprise On-Premises](#deployment-pattern-b-enterprise-on-premises)
5. [Voice Capture Endpoints](#voice-capture-endpoints)
6. [EHR Integration](#ehr-integration)
7. [Scaling Reference](#scaling-reference)
8. [High Availability](#high-availability)
9. [Security Considerations](#security-considerations)
10. [Cost Analysis](#cost-analysis)

---

## Overview

The voice-to-FHIR pipeline transforms spoken clinical documentation into structured FHIR resources through four stages:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Voice     │──▶│   MedASR    │──▶│  MedGemma   │──▶│    Post-    │──▶│    FHIR     │
│   Capture   │   │ Transcribe  │   │   Extract   │   │  Processor  │   │   Output    │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
     Audio            Text            Structured         Enriched          Resources
    (16kHz)        (Transcript)         JSON              JSON             (Bundle)
                                           │                │
                                           └────────────────┘
                                            • ICD-10 enrichment
                                            • RxNorm verification
                                            • Order-diagnosis linking
                                            • Validation & deduplication
```

### Design Principles

1. **Edge capture, centralized processing**: Voice captured at point of care, processed centrally
2. **Local transcription**: MedASR runs on-premises to minimize PHI transmission
3. **Flexible extraction**: MedGemma can be cloud or on-prem based on scale/compliance
4. **Standard integration**: FHIR R4 for EHR interoperability

---

## Component Architecture

### Core Components

| Component | Function | Resource Requirements |
|-----------|----------|----------------------|
| Voice Gateway | Receives audio from endpoints, routes to MedASR | 2 vCPU, 4GB RAM |
| MedASR Server | Speech-to-text transcription | NVIDIA T4 GPU, 16GB VRAM |
| MedGemma Server | Clinical entity extraction | NVIDIA L4 GPU, 24GB VRAM |
| FHIR Converter | Transforms JSON to FHIR resources | 2 vCPU, 4GB RAM |
| Integration Hub | Routes FHIR to EHR systems | 4 vCPU, 8GB RAM |

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLINICAL ENVIRONMENT                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                                   │
│  │ Mobile  │  │Workstation│ │ Voice   │                                   │
│  │   App   │  │  Client  │ │Assistant│                                   │
│  └────┬────┘  └────┬────┘  └────┬────┘                                   │
│       │            │            │                                         │
│       └────────────┼────────────┘                                         │
│                    │ Audio (TLS)                                          │
│                    ▼                                                      │
│  ┌─────────────────────────────────────────┐                             │
│  │            VOICE GATEWAY                 │                             │
│  │  • Audio validation                      │                             │
│  │  • Session management                    │                             │
│  │  • Queue management                      │                             │
│  └─────────────────┬───────────────────────┘                             │
│                    │                                                      │
│                    ▼                                                      │
│  ┌─────────────────────────────────────────┐                             │
│  │            MedASR CLUSTER               │  ◀── Always On-Premises     │
│  │  • Speech recognition                    │                             │
│  │  • Medical vocabulary                    │                             │
│  │  • GPU inference (T4)                    │                             │
│  └─────────────────┬───────────────────────┘                             │
│                    │ Transcript                                           │
└────────────────────┼─────────────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   ┌─────────────┐       ┌─────────────┐
   │  MedGemma   │       │  MedGemma   │
   │   Cloud     │       │  On-Prem    │
   │  (Pattern A)│       │ (Pattern B) │
   └──────┬──────┘       └──────┬──────┘
          │                     │
          └──────────┬──────────┘
                     │ Structured JSON
                     ▼
          ┌─────────────────────┐
          │   FHIR CONVERTER    │
          │  • Resource mapping │
          │  • Validation       │
          │  • Bundle creation  │
          └──────────┬──────────┘
                     │ FHIR Bundle
                     ▼
          ┌─────────────────────┐
          │   INTEGRATION HUB   │
          │  • Epic/Cerner API  │
          │  • HL7 v2 gateway   │
          │  • Direct FHIR push │
          └─────────────────────┘
```

---

## Deployment Pattern A: Small Practice

**Target**: 1-20 providers, 50-200 encounters/day

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PRACTICE NETWORK                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    VOICE CAPTURE TIER                         │   │
│  │                                                                │   │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │   │
│  │   │ iPhone  │   │  iPad   │   │Windows  │   │  Echo   │      │   │
│  │   │   App   │   │   App   │   │  Client │   │  Show   │      │   │
│  │   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘      │   │
│  │        └─────────────┴─────────────┴─────────────┘            │   │
│  │                          │ WebSocket/HTTPS                    │   │
│  └──────────────────────────┼───────────────────────────────────┘   │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    PROCESSING TIER                            │   │
│  │                                                                │   │
│  │   ┌────────────────────────────────────────────────────┐     │   │
│  │   │              SINGLE GPU SERVER                      │     │   │
│  │   │                                                      │     │   │
│  │   │   ┌─────────────┐     ┌─────────────────────┐       │     │   │
│  │   │   │   Voice     │     │      MedASR         │       │     │   │
│  │   │   │   Gateway   │────▶│   (NVIDIA T4 GPU)   │       │     │   │
│  │   │   │   :8080     │     │      :3002          │       │     │   │
│  │   │   └─────────────┘     └──────────┬──────────┘       │     │   │
│  │   │                                  │                   │     │   │
│  │   │   Hardware: Tower server with GPU support            │     │   │
│  │   │   • 1x NVIDIA T4 (16GB VRAM)                        │     │   │
│  │   │   • 32GB System RAM                                  │     │   │
│  │   │   • 500GB NVMe                                       │     │   │
│  │   └──────────────────────────┼───────────────────────────┘     │   │
│  │                              │                                  │   │
│  └──────────────────────────────┼─────────────────────────────────┘   │
│                                 │ Transcript (de-identified optional) │
└─────────────────────────────────┼─────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      INTERNET/VPN         │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CLOUD SERVICES                                  │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              HUGGINGFACE INFERENCE ENDPOINT                  │   │
│   │                                                               │   │
│   │   ┌─────────────────────────────────────────────────────┐   │   │
│   │   │                  MedGemma 4B                         │   │   │
│   │   │         Dedicated Endpoint (GPU Accelerated)         │   │   │
│   │   │                                                       │   │   │
│   │   │   • Model: google/medgemma-4b-it                     │   │   │
│   │   │   • Instance: nvidia-l4-x1                           │   │   │
│   │   │   • Scaling: 0-2 replicas (scale to zero)            │   │   │
│   │   │   • Region: us-east-1 (HIPAA eligible)               │   │   │
│   │   └─────────────────────────────────────────────────────┘   │   │
│   │                              │                               │   │
│   └──────────────────────────────┼──────────────────────────────┘   │
│                                  │                                   │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │ Structured JSON
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PRACTICE NETWORK                                │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   INTEGRATION TIER                           │   │
│   │                                                               │   │
│   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │   │
│   │   │    FHIR     │     │ Integration │     │    EHR      │   │   │
│   │   │  Converter  │────▶│     Hub     │────▶│   System    │   │   │
│   │   │             │     │             │     │             │   │   │
│   │   └─────────────┘     └─────────────┘     └─────────────┘   │   │
│   │                                                               │   │
│   │   Runs on: Existing server or VM (4 vCPU, 8GB RAM)          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Specifications

| Component | Hosting | Specifications | Est. Cost |
|-----------|---------|----------------|-----------|
| Voice Gateway | On-prem server | Docker container, 2 vCPU | Included |
| MedASR | On-prem GPU server | NVIDIA T4, 16GB VRAM | $4-6K (hardware) |
| MedGemma | HuggingFace Endpoint | L4 dedicated, scale-to-zero | $0.80/hr active |
| FHIR Converter | On-prem server | Docker container | Included |
| Integration Hub | On-prem server | Docker container | Included |

### Capacity Planning

With single T4 GPU running MedASR:
- **Throughput**: 15-20 transcriptions/hour (3-5 min recordings)
- **Concurrent users**: 5-8 (with queuing)
- **Daily capacity**: 120-160 encounters (8-hour day)

### Advantages

- Lower capital expenditure (single GPU server)
- Pay-per-use cloud extraction (scale to zero when idle)
- Simpler operations (fewer components to manage)
- PHI contained on-premises (only transcript leaves network)

### Considerations

- Internet dependency for MedGemma extraction
- Cloud egress costs for transcript data
- BAA required with HuggingFace for HIPAA compliance

---

## Deployment Pattern B: Enterprise On-Premises

**Target**: 50+ providers, 500+ encounters/day, hospital/health system

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HOSPITAL NETWORK                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         VOICE CAPTURE TIER                              │ │
│  │                                                                          │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │ │
│  │  │Provider │ │  Nurse  │ │  COW    │ │ Dictation│ │ Ambient │          │ │
│  │  │ Mobile  │ │ Station │ │Workstation││  Mic   │ │  Mic    │          │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │ │
│  │       └───────────┴───────────┴───────────┴───────────┘                │ │
│  │                               │                                         │ │
│  │                               │ Audio over TLS (mTLS for devices)       │ │
│  └───────────────────────────────┼─────────────────────────────────────────┘ │
│                                  ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       VOICE GATEWAY CLUSTER                             │ │
│  │                                                                          │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │                    LOAD BALANCER (F5/HAProxy)                    │  │ │
│  │   │                         voice.hospital.local                      │  │ │
│  │   └─────────────────────────────┬───────────────────────────────────┘  │ │
│  │                 ┌───────────────┼───────────────┐                      │ │
│  │                 ▼               ▼               ▼                      │ │
│  │          ┌───────────┐   ┌───────────┐   ┌───────────┐                │ │
│  │          │ Gateway 1 │   │ Gateway 2 │   │ Gateway 3 │                │ │
│  │          │  :8080    │   │  :8080    │   │  :8080    │                │ │
│  │          └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                │ │
│  │                └───────────────┼───────────────┘                       │ │
│  │                                │                                        │ │
│  │   Hardware: 3x VMs (4 vCPU, 8GB each) or Kubernetes pods              │ │
│  └────────────────────────────────┼────────────────────────────────────────┘ │
│                                   ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        MedASR GPU CLUSTER                               │ │
│  │                                                                          │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │              MESSAGE QUEUE (Redis/RabbitMQ)                      │  │ │
│  │   │                    Distributes audio jobs                        │  │ │
│  │   └─────────────────────────────┬───────────────────────────────────┘  │ │
│  │                 ┌───────────────┼───────────────┐                      │ │
│  │                 ▼               ▼               ▼                      │ │
│  │   ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐  │ │
│  │   │   MedASR Node 1   │ │   MedASR Node 2   │ │   MedASR Node 3   │  │ │
│  │   │                   │ │                   │ │                   │  │ │
│  │   │  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │  │ │
│  │   │  │  NVIDIA T4  │  │ │  │  NVIDIA T4  │  │ │  │  NVIDIA T4  │  │  │ │
│  │   │  │   16GB      │  │ │  │   16GB      │  │ │  │   16GB      │  │  │ │
│  │   │  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │  │ │
│  │   │      :3002        │ │      :3002        │ │      :3002        │  │ │
│  │   └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘  │ │
│  │             └───────────────┬─────┴───────────────┘                   │ │
│  │                             │                                          │ │
│  │   Hardware: 3x tower servers with single-slot GPU support             │ │
│  │   • 1x NVIDIA T4 per node (16GB VRAM)                                 │ │
│  │   • 32GB System RAM per node                                          │ │
│  │   • 500GB NVMe per node                                               │ │
│  └─────────────────────────────┼──────────────────────────────────────────┘ │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       MedGemma GPU CLUSTER                              │ │
│  │                                                                          │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │              MESSAGE QUEUE (shared or separate)                  │  │ │
│  │   │                 Distributes extraction jobs                      │  │ │
│  │   └─────────────────────────────┬───────────────────────────────────┘  │ │
│  │                 ┌───────────────┼───────────────┐                      │ │
│  │                 ▼               ▼               ▼                      │ │
│  │   ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐  │ │
│  │   │  MedGemma Node 1  │ │  MedGemma Node 2  │ │  MedGemma Node 3  │  │ │
│  │   │                   │ │                   │ │                   │  │ │
│  │   │  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │  │ │
│  │   │  │  NVIDIA L4  │  │ │  │  NVIDIA L4  │  │ │  │  NVIDIA L4  │  │  │ │
│  │   │  │   24GB      │  │ │  │   24GB      │  │ │  │   24GB      │  │  │ │
│  │   │  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │  │ │
│  │   │      :3003        │ │      :3003        │ │      :3003        │  │ │
│  │   └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘  │ │
│  │             └───────────────┬─────┴───────────────┘                   │ │
│  │                             │                                          │ │
│  │   Hardware: 3x 2U rack servers with GPU support                       │ │
│  │   Alternative: 2x NVIDIA A100 (80GB) for higher throughput            │ │
│  └─────────────────────────────┼──────────────────────────────────────────┘ │
│                                ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        INTEGRATION TIER                                 │ │
│  │                                                                          │ │
│  │   ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │ │
│  │   │     FHIR      │    │  Integration  │    │      EHR SYSTEMS      │ │ │
│  │   │   Converter   │───▶│      Hub      │───▶│                       │ │ │
│  │   │   Cluster     │    │   (Mirth/     │    │  ┌─────┐   ┌─────┐   │ │ │
│  │   │               │    │   Rhapsody)   │    │  │Epic │   │Cerner│   │ │ │
│  │   └───────────────┘    └───────────────┘    │  └─────┘   └─────┘   │ │ │
│  │                                              │                       │ │ │
│  │   Hardware: 3x VMs (4 vCPU, 8GB) or K8s pods │  ┌─────┐   ┌─────┐   │ │ │
│  │                                              │  │MEDITECH │Allscripts││ │ │
│  └──────────────────────────────────────────────│  └─────┘   └─────┘   │─┘ │
│                                                 └───────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Standalone vs. Cluster Deployment

#### Standalone (Single Multi-GPU Server)

For mid-size deployments (200-400 encounters/day):

```
┌─────────────────────────────────────────────────────────┐
│              NVIDIA DGX STATION A100                     │
│                                                          │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│   │  GPU 0    │  │  GPU 1    │  │  GPU 2    │          │
│   │  MedASR   │  │  MedASR   │  │ MedGemma  │          │
│   │  (T4)     │  │  (T4)     │  │   (L4)    │          │
│   └───────────┘  └───────────┘  └───────────┘          │
│                                                          │
│   • 2x NVIDIA T4 16GB (MedASR) + 2x NVIDIA L4 24GB (MedGemma) │
│   • 512GB System RAM                                     │
│   • 8TB NVMe RAID                                        │
│   • Dual 10GbE                                           │
│                                                          │
│   Capacity: ~60 transcriptions/hour                      │
└─────────────────────────────────────────────────────────┘
```

#### Cluster (Distributed Processing)

For large deployments (500+ encounters/day):

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KUBERNETES CLUSTER                               │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    CONTROL PLANE                             │   │
│   │   etcd, API Server, Scheduler, Controller Manager            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│   │  GPU Node 1 │  │  GPU Node 2 │  │  GPU Node 3 │  ...          │
│   │   (T4)      │  │   (T4)      │  │   (L4)      │               │
│   │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │               │
│   │  │ T4 GPU│  │  │  │ T4 GPU│  │  │  │ L4 GPU│  │               │
│   │  └───────┘  │  │  └───────┘  │  │  └───────┘  │               │
│   │             │  │             │  │             │               │
│   │  Pods:      │  │  Pods:      │  │  Pods:      │               │
│   │  - medasr   │  │  - medasr   │  │  - medgemma │               │
│   │  - gateway  │  │  - medgemma │  │  - medgemma │               │
│   └─────────────┘  └─────────────┘  └─────────────┘               │
│                                                                      │
│   Autoscaling: HPA based on queue depth                             │
│   Node pools: medasr-pool (T4), medgemma-pool (L4)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Specifications

| Component | Nodes | Specifications | Est. Cost |
|-----------|-------|----------------|-----------|
| Voice Gateway | 3 | 4 vCPU, 8GB RAM each | $3K (VMs) |
| MedASR Cluster | 3 | NVIDIA T4, 16GB VRAM each | $12-15K |
| MedGemma Cluster | 3 | NVIDIA L4, 24GB VRAM each | $24-30K |
| Message Queue | 3 | Redis Cluster or RabbitMQ | $3K |
| FHIR Converter | 3 | 4 vCPU, 8GB RAM each | $3K |
| Integration Hub | 2 | Mirth Connect or Rhapsody | License varies |

### Capacity Planning

With 3-node MedASR cluster (T4) + 3-node MedGemma cluster (L4):
- **Throughput**: 60-80 transcriptions/hour
- **Concurrent users**: 30-50
- **Daily capacity**: 480-640 encounters (8-hour day)
- **Peak handling**: Queue absorbs bursts up to 100 concurrent

---

## Voice Capture Endpoints

### Supported Devices

| Device Type | Integration Method | Audio Format | Notes |
|-------------|-------------------|--------------|-------|
| iOS (iPhone/iPad) | Native app + WebSocket | 16kHz PCM | Background recording support |
| Android | Native app + WebSocket | 16kHz PCM | Foreground service required |
| Windows Workstation | Desktop client | 16kHz PCM | Hotkey activation |
| macOS Workstation | Desktop client | 16kHz PCM | Accessibility permissions |
| Web Browser | WebRTC/MediaRecorder | 16kHz WebM/Opus | Chrome, Edge, Firefox |
| Amazon Echo Show | Alexa Skill | 16kHz PCM | "Alexa, start clinical note" |
| Dedicated Microphone | USB audio class | 16kHz PCM | Dragon-compatible devices |

### Voice Gateway Protocol

```
┌──────────────────────────────────────────────────────────────────────┐
│                     VOICE CAPTURE PROTOCOL                            │
│                                                                        │
│   1. Authentication                                                    │
│      Client ──▶ Gateway: POST /auth {provider_id, device_id, token}   │
│      Gateway ──▶ Client: {session_id, websocket_url}                  │
│                                                                        │
│   2. Session Start                                                     │
│      Client ──▶ Gateway: WS CONNECT /audio/{session_id}               │
│      Client ──▶ Gateway: {type: "start", patient_id, workflow}        │
│                                                                        │
│   3. Audio Streaming                                                   │
│      Client ──▶ Gateway: Binary frames (16kHz PCM, 100ms chunks)      │
│      Gateway ──▶ Client: {type: "ack", bytes_received}                │
│                                                                        │
│   4. Session End                                                       │
│      Client ──▶ Gateway: {type: "stop"}                               │
│      Gateway ──▶ Client: {type: "processing", job_id}                 │
│                                                                        │
│   5. Result Delivery                                                   │
│      Gateway ──▶ Client: {type: "complete", fhir_bundle, transcript}  │
│      -or-                                                              │
│      Client ──▶ Gateway: GET /results/{job_id} (polling)              │
└──────────────────────────────────────────────────────────────────────┘
```

### Mobile App Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      iOS/ANDROID APP                             │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                    UI LAYER                                │ │
│   │   • Recording controls (push-to-talk, continuous)         │ │
│   │   • Waveform visualization                                 │ │
│   │   • Patient context display                                │ │
│   │   • Result review/edit                                     │ │
│   └───────────────────────────────────────────────────────────┘ │
│                              │                                   │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                   AUDIO ENGINE                             │ │
│   │   • Microphone capture (AVAudioEngine / AudioRecord)      │ │
│   │   • Resampling to 16kHz                                    │ │
│   │   • VAD (Voice Activity Detection)                         │ │
│   │   • Noise suppression (optional)                           │ │
│   │   • Local buffer (offline support)                         │ │
│   └───────────────────────────────────────────────────────────┘ │
│                              │                                   │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                 NETWORK LAYER                              │ │
│   │   • WebSocket connection management                        │ │
│   │   • Automatic reconnection                                 │ │
│   │   • Offline queue with sync                                │ │
│   │   • Certificate pinning                                    │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## EHR Integration

### Integration Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EHR INTEGRATION OPTIONS                           │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  PATTERN 1: FHIR R4 Native                                       │   │
│   │                                                                   │   │
│   │   FHIR Bundle ──▶ EHR FHIR Server ──▶ Clinical Documentation     │   │
│   │                                                                   │   │
│   │   Supported by: Epic (2020+), Cerner Millennium, Oracle Health   │   │
│   │   Resources: DocumentReference, DiagnosticReport, Observation    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  PATTERN 2: CDA Document Exchange                                │   │
│   │                                                                   │   │
│   │   FHIR Bundle ──▶ FHIR-to-CDA ──▶ CDA Document ──▶ EHR          │   │
│   │                   Converter                                       │   │
│   │                                                                   │   │
│   │   Supported by: Most EHRs via document import                    │   │
│   │   Standard: C-CDA R2.1                                           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  PATTERN 3: HL7 v2 ADT/ORU                                       │   │
│   │                                                                   │   │
│   │   FHIR Bundle ──▶ FHIR-to-HL7v2 ──▶ HL7 Message ──▶ EHR         │   │
│   │                   Converter            Interface Engine          │   │
│   │                                                                   │   │
│   │   Supported by: Legacy EHRs, lab systems                         │   │
│   │   Messages: ORU^R01 (results), MDM^T02 (documents)               │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  PATTERN 4: Vendor-Specific API                                  │   │
│   │                                                                   │   │
│   │   FHIR Bundle ──▶ Custom Adapter ──▶ Vendor API ──▶ EHR         │   │
│   │                                                                   │   │
│   │   Examples:                                                       │   │
│   │   • Epic: MyChart Bedside API, Interconnect                      │   │
│   │   • Cerner: Millennium Objects API                               │   │
│   │   • Allscripts: Unity API                                        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Integration Hub Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION HUB                                  │
│                    (Mirth Connect / Rhapsody)                           │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      INBOUND CHANNEL                             │   │
│   │   Source: FHIR Converter REST endpoint                          │   │
│   │   Format: FHIR R4 Bundle                                         │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                       ROUTER                                     │   │
│   │   Routes based on: patient MRN prefix, encounter type,          │   │
│   │                    destination system availability               │   │
│   └───────────────────────────────┬─────────────────────────────────┘   │
│               ┌───────────────────┼───────────────────┐                 │
│               ▼                   ▼                   ▼                 │
│   ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐   │
│   │  EPIC CHANNEL     │ │  CERNER CHANNEL   │ │  LEGACY CHANNEL   │   │
│   │                   │ │                   │ │                   │   │
│   │  Transform:       │ │  Transform:       │ │  Transform:       │   │
│   │  FHIR → Epic API  │ │  FHIR → Cerner API│ │  FHIR → HL7 v2    │   │
│   │                   │ │                   │ │                   │   │
│   │  Destination:     │ │  Destination:     │ │  Destination:     │   │
│   │  epic.hospital/   │ │  cerner.hospital/ │ │  TCP 2575         │   │
│   │  api/FHIR/R4      │ │  api/FHIR/R4      │ │  (MLLP)           │   │
│   └───────────────────┘ └───────────────────┘ └───────────────────┘   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    ERROR HANDLING                                │   │
│   │   • Dead letter queue for failed messages                       │   │
│   │   • Automatic retry with exponential backoff                    │   │
│   │   • Alert on repeated failures                                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### FHIR Resource Mapping

| Extracted Entity | FHIR Resource | Key Fields |
|------------------|---------------|------------|
| Patient demographics | Patient | name, birthDate, gender |
| Conditions/Diagnoses | Condition | code (ICD-10), clinicalStatus |
| Medications | MedicationStatement | medicationCodeableConcept, dosage |
| Medication orders | MedicationRequest | medicationCodeableConcept, dosageInstruction |
| Allergies | AllergyIntolerance | code, reaction, criticality |
| Vitals | Observation | code (LOINC), value, effectiveDateTime |
| Lab results | Observation | code (LOINC), value, interpretation |
| Lab orders | ServiceRequest | code (LOINC), intent=order |
| Procedure orders | ServiceRequest | code (CPT/SNOMED), intent=order |
| Referrals | ServiceRequest | code, intent=order, performerType |
| Family history | FamilyMemberHistory | relationship, condition |
| Social history | Observation | code (social-history category) |
| Full transcript | DocumentReference | content (base64), type=clinical-note |

---

## Clinical Decision Support Post-Processing

After MedGemma extraction, the post-processor enriches data with clinical terminology and decision support:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POST-PROCESSING PIPELINE                                │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 1: Validation & Deduplication                                 │   │
│   │                                                                       │   │
│   │  • Remove placeholder values ("null", "unknown", "n/a")              │   │
│   │  • Deduplicate medications, conditions, vitals                       │   │
│   │  • Filter invalid entries (non-medications in orders)                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 2: ICD-10 Code Enrichment                                     │   │
│   │                                                                       │   │
│   │  • 500+ condition → ICD-10-CM code mappings                          │   │
│   │  • Synonym matching ("heart attack" → I21.9)                         │   │
│   │  • Fuzzy matching for spelling variations (85% threshold)            │   │
│   │  • Confidence scoring                                                 │   │
│   │                                                                       │   │
│   │  Data source: CMS ICD-10-CM (public domain)                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 3: RxNorm Medication Verification                             │   │
│   │                                                                       │   │
│   │  • 200+ common medications with RxCUI codes                          │   │
│   │  • Brand-to-generic mapping (Lipitor → atorvastatin)                 │   │
│   │  • Drug class identification (atorvastatin → statin)                 │   │
│   │  • Fuzzy matching for transcription errors                           │   │
│   │  • Unverified medications flagged for clinician review               │   │
│   │                                                                       │   │
│   │  Data source: NLM RxNorm (UMLS license)                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 4: Order-Diagnosis Linking                                    │   │
│   │                                                                       │   │
│   │  Links orders to diagnoses using clinical rules:                     │   │
│   │                                                                       │   │
│   │  • Medication orders → Drug class → Typical indications              │   │
│   │    (atorvastatin → statin → E78.5 Hyperlipidemia)                    │   │
│   │                                                                       │   │
│   │  • Lab orders → Test → Monitoring indications                        │   │
│   │    (HbA1c → E11.9 Type 2 diabetes)                                   │   │
│   │                                                                       │   │
│   │  • Consult orders → Specialty → Typical conditions                   │   │
│   │    (Cardiology → I25.10 CAD, I50.9 Heart failure)                    │   │
│   │                                                                       │   │
│   │  • Procedure orders → Procedure → Typical indications                │   │
│   │    (Echocardiogram → I50.9 Heart failure)                            │   │
│   │                                                                       │   │
│   │  Priority: Patient conditions matched first, then rule defaults      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Clinical Rule Coverage

| Order Type | Rule Count | Examples |
|------------|------------|----------|
| Medications | 40+ drug classes | Statins → hyperlipidemia, ACE-I → HTN/CHF, SSRIs → depression |
| Labs | 50+ tests | Lipid panel → hyperlipidemia, TSH → thyroid disorders, BNP → CHF |
| Consults | 30+ specialties | Cardiology → CAD/CHF, Nephrology → CKD, Psychiatry → depression |
| Procedures | 50+ procedures | ECG → arrhythmia, Stress test → CAD, Colonoscopy → screening |

### Data Sources

| Dataset | Source | License | Usage |
|---------|--------|---------|-------|
| ICD-10-CM codes | CMS | Public domain | Condition coding |
| RxNorm medications | NLM | UMLS (free for US) | Medication verification |
| LOINC codes | Regenstrief | Free with attribution | Lab test coding |
| Clinical rules | Cleansheet | CC BY 4.0 | Order-diagnosis linking |

---

## Scaling Reference

### GPU Performance Benchmarks

**MedASR on NVIDIA T4 (16GB)**

| Input Size | Latency (p50) | Latency (p99) | Throughput |
|------------|---------------|---------------|------------|
| 1 min audio | 10-15s | 22s | 4-5/min |
| 3 min audio | 30-40s | 60s | 1.5-2/min |
| 5 min audio | 55-75s | 110s | 0.8-1/min |
| 8 min audio | 100-140s | 200s | 0.4-0.5/min |

*T4 is ~15-20% slower than L4 for MedASR, but sufficient for most clinical workloads.*

**MedGemma 4B on NVIDIA L4 (24GB)**

| Input Size | Latency (p50) | Latency (p99) | Throughput |
|------------|---------------|---------------|------------|
| 500 tokens | 3-5s | 8s | 12-15/min |
| 2000 tokens | 10-15s | 25s | 4-5/min |
| 8000 tokens | 40-60s | 90s | 1/min |

### Cluster Sizing Guide

| Daily Volume | MedASR Nodes (T4) | MedGemma Nodes (L4) | Total GPUs |
|--------------|-------------------|---------------------|------------|
| 50-100 encounters | 1 | 1 (or cloud) | 1-2 |
| 100-200 encounters | 1 | 1 | 2 |
| 200-400 encounters | 2 | 2 | 4 |
| 400-600 encounters | 3 | 3 | 6 |
| 600-1000 encounters | 4-5 | 4-5 | 8-10 |
| 1000+ encounters | 6+ | 6+ | 12+ |

### Queue Depth Recommendations

| Queue Depth | Action |
|-------------|--------|
| < 10 | Normal operation |
| 10-25 | Consider adding nodes |
| 25-50 | Scale up urgently |
| > 50 | Reject new requests, alert ops |

---

## High Availability

### Pattern A: Small Practice HA

```
┌─────────────────────────────────────────────────────────────────┐
│                    WARM STANDBY                                  │
│                                                                  │
│   ┌─────────────────────┐     ┌─────────────────────┐          │
│   │   PRIMARY SERVER    │     │   STANDBY SERVER    │          │
│   │                     │     │                     │          │
│   │  ┌───────────────┐  │     │  ┌───────────────┐  │          │
│   │  │    MedASR     │  │     │  │    MedASR     │  │          │
│   │  │   (Active)    │  │     │  │  (Passive)    │  │          │
│   │  └───────────────┘  │     │  └───────────────┘  │          │
│   │                     │     │                     │          │
│   │  Heartbeat ─────────┼─────┼──▶ Monitoring       │          │
│   │                     │     │                     │          │
│   └─────────────────────┘     └─────────────────────┘          │
│                                                                  │
│   Failover: DNS update or floating IP (manual or scripted)      │
│   RTO: 5-15 minutes                                              │
│   RPO: 0 (no state, audio reprocessed from client buffer)       │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern B: Enterprise HA

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       ACTIVE-ACTIVE CLUSTER                              │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                    LOAD BALANCER (ACTIVE-ACTIVE)                   │ │
│   │              F5 BIG-IP or HAProxy with keepalived                  │ │
│   │                                                                     │ │
│   │   VIP: voice.hospital.local                                        │ │
│   │   Health checks: /health every 5s                                  │ │
│   │   Failover: < 30 seconds                                           │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│            ┌─────────────────┼─────────────────┐                        │
│            ▼                 ▼                 ▼                        │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │
│   │   MedASR Pod 1  │ │   MedASR Pod 2  │ │   MedASR Pod 3  │          │
│   │   (Active)      │ │   (Active)      │ │   (Active)      │          │
│   └─────────────────┘ └─────────────────┘ └─────────────────┘          │
│                                                                          │
│   Queue: Redis Sentinel (3 nodes) or Redis Cluster                      │
│   Replication: Automatic failover with Sentinel                         │
│                                                                          │
│   RTO: < 30 seconds                                                     │
│   RPO: 0 (jobs in queue preserved)                                      │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                    DISASTER RECOVERY                               │ │
│   │                                                                     │ │
│   │   Secondary Site: Replicated GPU cluster in DR datacenter          │ │
│   │   Replication: Queue mirroring (Redis cross-datacenter)            │ │
│   │   Failover: DNS GSLB or manual cutover                             │ │
│   │   RTO: 15-30 minutes                                                │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Single GPU node failure | Reduced throughput | Queue redistributes to remaining nodes |
| Queue failure | Job loss | Redis Sentinel automatic failover |
| Gateway failure | Dropped connections | LB health checks, client reconnect |
| Full cluster failure | Service unavailable | DR site activation |
| Cloud endpoint down (Pattern A) | Extraction unavailable | Queue locally, retry when restored |

---

## Security Considerations

### Data Flow Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA CLASSIFICATION                                 │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │  ZONE 1: PHI (Protected Health Information)                     │    │
│   │                                                                  │    │
│   │  • Raw audio (contains patient identifiers in speech)           │    │
│   │  • Transcripts (full clinical content)                          │    │
│   │  • FHIR bundles (structured patient data)                       │    │
│   │                                                                  │    │
│   │  Controls: Encryption at rest (AES-256), TLS 1.3 in transit,   │    │
│   │            access logging, minimum necessary principle          │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │  ZONE 2: De-identified (optional for cloud extraction)          │    │
│   │                                                                  │    │
│   │  • Transcripts with patient identifiers removed                 │    │
│   │  • Dates shifted, ages generalized (HIPAA Safe Harbor)          │    │
│   │                                                                  │    │
│   │  Controls: De-identification pipeline, audit trail              │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │  ZONE 3: Operational (no PHI)                                   │    │
│   │                                                                  │    │
│   │  • Job metadata (IDs, timestamps, status)                       │    │
│   │  • Performance metrics                                           │    │
│   │  • Error logs (sanitized)                                        │    │
│   │                                                                  │    │
│   │  Controls: Standard security controls                           │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Network Segmentation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NETWORK ZONES                                       │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  DMZ                                                             │   │
│   │  • Voice Gateway (accepts external connections)                 │   │
│   │  • TLS termination                                               │   │
│   │  • WAF protection                                                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │ (Firewall)                               │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Processing Zone                                                 │   │
│   │  • MedASR cluster (no external access)                          │   │
│   │  • MedGemma cluster (no external access, or outbound to cloud)  │   │
│   │  • Message queue                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │ (Firewall)                               │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Integration Zone                                                │   │
│   │  • FHIR Converter                                                │   │
│   │  • Integration Hub                                               │   │
│   │  • EHR connectivity                                              │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Compliance Mapping

| Requirement | HIPAA | SOC2 | FDA 510(k) |
|-------------|-------|------|------------|
| Encryption at rest | § 164.312(a)(2)(iv) | CC6.1 | Cybersecurity |
| Encryption in transit | § 164.312(e)(1) | CC6.6 | Cybersecurity |
| Access controls | § 164.312(a)(1) | CC6.1-6.3 | Access control |
| Audit logging | § 164.312(b) | CC7.2 | Audit trail |
| Integrity controls | § 164.312(c)(1) | CC6.1 | Data integrity |
| BAA with cloud vendors | § 164.502(e) | N/A | N/A |

---

## Cost Analysis

### Pattern A: Small Practice (Monthly)

| Component | Capital | Monthly Operating |
|-----------|---------|-------------------|
| GPU Server (1x T4 for MedASR) | $5,000 | $75 (power/cooling) |
| HuggingFace Endpoint (MedGemma) | $0 | $200-500 (usage-based) |
| Network/Infrastructure | $2,000 | $50 |
| **Total** | **$7,000** | **$325-625/mo** |

**3-Year TCO**: $7,000 + ($475 × 36) = **$24,100**

### Pattern B: Enterprise On-Prem (Monthly)

| Component | Capital | Monthly Operating |
|-----------|---------|-------------------|
| MedASR Servers (3x T4) | $15,000 | $225 (power/cooling) |
| MedGemma Servers (3x L4) | $30,000 | $225 (power/cooling) |
| Kubernetes Infrastructure | $15,000 | $200 |
| Integration Hub (Mirth) | $0-25,000 | $0-500 (support) |
| Network/Storage | $10,000 | $150 |
| **Total** | **$70,000-95,000** | **$800-1,300/mo** |

**3-Year TCO**: $82,500 + ($1,050 × 36) = **$120,300**

### Cloud-Only Alternative (Reference)

| Component | Monthly Cost |
|-----------|--------------|
| MedASR (HF Dedicated) | $1,200-2,400 |
| MedGemma (HF Dedicated) | $1,200-2,400 |
| Gateway (Cloud VM) | $200 |
| **Total** | **$2,600-5,000/mo** |

**3-Year TCO**: $3,800 × 36 = **$136,800**

*Cloud-only has lower capital but higher operating costs; breaks even with on-prem at ~3 years.*

---

## Appendix: Hardware Specifications

### Recommended Server Configurations

**Small Practice - MedASR Server (Pattern A)**
```
Tower server with single-slot GPU support
├── CPU: x86-64, 16+ cores (e.g., Xeon Silver class)
├── RAM: 32GB DDR5
├── GPU: 1x NVIDIA T4 16GB
├── Storage: 2x 480GB NVMe RAID 1
├── Network: 2x 1GbE
└── Power: 600W PSU
```

**Enterprise MedASR Node (Pattern B)**
```
Tower server with single-slot GPU support
├── CPU: x86-64, 16+ cores (e.g., Xeon Silver class)
├── RAM: 32GB DDR5
├── GPU: 1x NVIDIA T4 16GB
├── Storage: 500GB NVMe
├── Network: 2x 10GbE
└── Power: 600W PSU
```

**Enterprise MedGemma Node (Pattern B)**
```
2U rack server with full-height GPU support
├── CPU: x86-64, 32+ cores (e.g., Xeon Gold class)
├── RAM: 64GB DDR5
├── GPU: 1x NVIDIA L4 24GB
├── Storage: 2x 960GB NVMe RAID 1
├── Network: 2x 10GbE
└── Power: 2x 800W redundant PSU
```

### GPU Comparison

| GPU | VRAM | FP16 TFLOPS | Power | Est. Price | Best For |
|-----|------|-------------|-------|------------|----------|
| NVIDIA T4 | 16GB | 65 | 70W | $1,000 | **MedASR** (cost-optimized) |
| NVIDIA L4 | 24GB | 120 | 72W | $2,500 | **MedGemma** (recommended) |
| NVIDIA A10 | 24GB | 125 | 150W | $4,000 | Higher throughput |
| NVIDIA A100 40GB | 40GB | 312 | 250W | $10,000 | Large models |
| NVIDIA A100 80GB | 80GB | 312 | 300W | $15,000 | Maximum capacity |

---

*Last updated: 2026-01-28*
