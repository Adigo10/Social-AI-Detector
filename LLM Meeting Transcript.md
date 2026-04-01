### **Project Status Update**

* Embedding generation with Gemini API 60% complete  
  1. Started 3 hours ago, 3 more hours remaining (completion \~9pm)  
  2. Using free Google credits  
  3. Will take total 6 hours for full dataset  
* Next steps after embedding completion:  
  1. Build 5 indices  
  2. Training data generation  
  3. Scenario-specific index builds (normal, cross-model, cross-platform scenarios)  
* All data to be uploaded to Google Drive by tomorrow

### **Model Comparison Strategy**

* Finalized approach: Two different instruct models (not parameter variations)  
  1. Model 1: Qwen 2.5 instruct  
  2. Model 2: DeepSeek or similar \~30-32B parameter model  
* Six core tasks across both models:  
  1. Zero-shot baseline (no training required)  
  2. Fine-tuning with RAG data  
  3. Fine-tuning without RAG data  
* Seventh comparison: Commercial model (GPT-4o or similar) for inference-only benchmark

### **Technical Implementation**

* Switch from majority voting to weighted voting using similarity scores  
  * Draft branch already available for review  
* Use QLora for all fine-tuning (industry standard, efficient)  
* Weights & Biases integration for ablation studies  
  * Team account setup needed for shared access  
  * GitHub repository linking to be investigated

### **Task Assignments**

* Qwen instruct model: Sneha \+ Ananya  
* Second model (DeepSeek/alternative): Twissa \+ Tanmay  
* RAG \+ fine-tuning ensemble design: Aditya \+ Yu Taek  
* Evaluation and ablation studies: Existing team member continuing current work  
* Parallel coordination on parameter tuning between assigned pairs

### **Timeline & Deliverables**

* Hard deadline: All tasks complete by next Tuesday  
* Video submission: Next Friday (before April 13th final submission)  
* Daily progress updates via group chat  
* Embedding completion notification when ready  
* Model recommendations to be shared in WhatsApp group

