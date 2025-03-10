# **HackScript 6.0. Hackathon** 

## Team Members  
**Vedant Mallya**  
**Sayed Iqra**  
**Pranav Ghodke**  
**Vivek Maurya**  

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# **QA-BOT: Real-Time Performance Analysis for Voice Agents**

## Description

QA-BOT is a real-time analytical tool designed to evaluate the performance of voice agents by processing their interactions with customers. The system extracts key performance metrics such as **accuracy, sentiment, and responsiveness**, providing **actionable insights** to enhance customer service quality.

## Features

- **Real-Time Analysis:** Processes live audio data from customer interactions.  
- **Sentiment Detection:** Identifies positive, neutral, and negative sentiments.  
- **Performance Metrics Evaluation:** Measures response time, tone, and accuracy.  
- **Automated Alerts:** Flags critical issues like prolonged silence or negative sentiment.  
- **Actionable Insights:** Provides recommendations for improving voice agent performance.  

## Required Solution

### Input:
- **Real-time audio data** from voice agents’ customer interactions.  
- **Call transcripts and metadata** (e.g., duration, sentiment, keywords).  
- **Predefined performance metrics** (e.g., response time, tone, accuracy).  

### Output:
- **Performance scores and insights** for each voice agent.  
- **Alerts for critical issues** (e.g., negative comments, long pauses). 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Repository Specification**  

### **Database Schema**  

To set up the database, execute the following SQL commands:

```sql
-- Create the database
CREATE DATABASE qa1;
USE qa1;

-- Create table to store conversation metrics
CREATE TABLE conversation_metrics1 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    accuracy FLOAT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### **Admin Panel**  

Admin name: admin 

Password: admin123

