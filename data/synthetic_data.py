from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from core.exceptions import DataGenerationError

CLIENT_PROFILES = [
    {
        "client_id": "client_1",
        "domain": "Healthcare",
        "language": "Hindi",
        "description": "Rural healthcare assistant for Hindi-speaking users",
    },
    {
        "client_id": "client_2",
        "domain": "Education",
        "language": "Marathi",
        "description": "School education support for Marathi-speaking students",
    },
    {
        "client_id": "client_3",
        "domain": "Financial Literacy",
        "language": "Tamil",
        "description": "Financial literacy guidance for Tamil-speaking users",
    },
]

_HEALTHCARE_SAMPLES = [
    {"input": "बुखार होने पर क्या करें?", "output": "अधिक पानी पिएं, आराम करें और पेरासिटामोल लें। यदि बुखार 3 दिन से अधिक रहे तो डॉक्टर से मिलें।"},
    {"input": "मधुमेह रोगी को क्या खाना चाहिए?", "output": "कम चीनी, अधिक फाइबर युक्त भोजन जैसे सब्जियां, दालें और साबुत अनाज खाएं।"},
    {"input": "रक्तचाप को कैसे नियंत्रित करें?", "output": "नमक कम खाएं, नियमित व्यायाम करें, तनाव कम लें और डॉक्टर की सलाह से दवाएं लें।"},
    {"input": "बच्चे को टीका कब लगवाएं?", "output": "जन्म के बाद पोलियो, BCG, हेपेटाइटिस B टीके लगवाएं। सरकारी टीकाकरण कार्यक्रम का पालन करें।"},
    {"input": "खांसी का घरेलू उपाय क्या है?", "output": "गर्म पानी में शहद और अदरक मिलाकर पिएं। गर्म भाप लें और गले को गर्म रखें।"},
    {"input": "एनीमिया के लक्षण क्या हैं?", "output": "कमजोरी, थकान, पीली त्वचा और सांस लेने में कठिनाई एनीमिया के सामान्य लक्षण हैं।"},
    {"input": "गर्भावस्था में क्या सावधानियां रखें?", "output": "नियमित जांच कराएं, पोषक भोजन खाएं, भारी काम से बचें और डॉक्टर की सलाह मानें।"},
    {"input": "मलेरिया से कैसे बचें?", "output": "मच्छरदानी का उपयोग करें, पानी जमा न होने दें और मच्छर भगाने वाली क्रीम लगाएं।"},
    {"input": "पेट दर्द होने पर क्या करें?", "output": "हल्का गर्म पानी पिएं, तला-भुना न खाएं। यदि दर्द तेज हो या लंबे समय तक रहे तो डॉक्टर को दिखाएं।"},
    {"input": "नींद न आने की समस्या का समाधान?", "output": "सोने से पहले मोबाइल न देखें, एक निश्चित समय पर सोएं और कैफीन से बचें।"},
]

_EDUCATION_SAMPLES = [
    {"input": "गणित शिकणे सोपे कसे करायचे?", "output": "रोज थोडा सराव करा, उदाहरणे सोडवा आणि कठीण संकल्पना शिक्षकांना विचारा."},
    {"input": "परीक्षेची तयारी कशी करावी?", "output": "वेळापत्रक बनवा, नोट्स काढा, मागील प्रश्नपत्रिका सोडवा आणि पुरेशी झोप घ्या."},
    {"input": "इंग्रजी भाषा सुधारण्यासाठी काय करावे?", "output": "दररोज इंग्रजी वाचा, लिहा आणि बोलण्याचा प्रयत्न करा. चित्रपट पाहणे आणि पुस्तके वाचणे फायदेशीर आहे."},
    {"input": "विज्ञान प्रयोग कसे समजावून घ्यावेत?", "output": "प्रत्यक्ष प्रयोग करा, निरीक्षणे नोंदवा आणि परिणाम समजून घेण्याचा प्रयत्न करा."},
    {"input": "शाळेत लक्ष केंद्रित कसे करावे?", "output": "पुरेशी झोप घ्या, सकाळी नाश्ता करा, वर्गात प्रश्न विचारा आणि मोबाइलपासून दूर राहा."},
    {"input": "इतिहास लक्षात कसा ठेवावा?", "output": "घटनांच्या कथा स्वरूपात वाचा, टाइमलाइन बनवा आणि महत्त्वाच्या तारखा लिहून ठेवा."},
    {"input": "संगणक शिकणे कुठून सुरू करावे?", "output": "टायपिंग, MS Office, आणि इंटरनेट वापरापासून सुरुवात करा. नंतर कोडिंग शिकता येईल."},
    {"input": "शिष्यवृत्ती कशी मिळवावी?", "output": "शाळेच्या परिणामात चांगले गुण मिळवा, सरकारी योजनांची माहिती घ्या आणि वेळेवर अर्ज करा."},
    {"input": "वाचन कौशल्य कसे सुधारावे?", "output": "दररोज थोडे वाचा, अवघड शब्दांचे अर्थ समजून घ्या आणि वाचलेल्या गोष्टींबद्दल विचार करा."},
    {"input": "गृहपाठ वेळेवर कसा पूर्ण करावा?", "output": "शाळेतून आल्यावर थोडी विश्रांती घ्या, मग एकाग्रतेने गृहपाठ करा आणि नंतर खेळा."},
]

_FINANCIAL_SAMPLES = [
    {"input": "சேமிப்பை எப்படி தொடங்குவது?", "output": "மாதாமாதம் வருமானத்தில் குறைந்தது 20% சேமிக்கவும். வங்கி கணக்கு திறந்து தானியங்கி சேமிப்பை அமைக்கவும்."},
    {"input": "கடனை எப்படி தவிர்ப்பது?", "output": "தேவையில்லாத செலவுகளை குறைக்கவும், கடன் அட்டை பயன்படுத்துவதை கட்டுப்படுத்தவும் மற்றும் அவசர நிதி வைத்திருக்கவும்."},
    {"input": "முதலீடு எங்கே செய்வது நல்லது?", "output": "PPF, FD அல்லது மியூச்சுவல் ஃபண்டுகளில் முதலீடு செய்யலாம். நிதி ஆலோசகரிடம் அறிவுரை பெறவும்."},
    {"input": "வரி செலுத்துவது எப்படி?", "output": "வருடாந்திர வருமானம் கணக்கிடுவும், ITR படிவம் நிரப்பவும் மற்றும் காலக்கெடுவிற்கு முன் செலுத்தவும்."},
    {"input": "காப்பீடு ஏன் முக்கியம்?", "output": "உடல் நலம், வாகனம் மற்றும் உயிர் காப்பீடு எதிர்பாராத நிதி நெருக்கடிகளில் இருந்து பாதுகாக்கும்."},
    {"input": "வங்கி கணக்கு எப்படி நிர்வகிப்பது?", "output": "மாதாமாதம் கணக்கு சரிபார்க்கவும், தேவையற்ற கட்டணங்கள் இல்லாத திட்டங்களை தேர்வு செய்யவும்."},
    {"input": "SIP என்றால் என்ன?", "output": "SIP என்பது Systematic Investment Plan. ஒவ்வொரு மாதமும் கொஞ்சம் கொஞ்சமாக மியூச்சுவல் ஃபண்டில் முதலீடு செய்யும் முறை."},
    {"input": "அடமான கடன் எப்படி பெறுவது?", "output": "நல்ல கடன் மதிப்பெண் பராமரிக்கவும், வருமான ஆவணங்கள் தயாரிக்கவும் மற்றும் வங்கிகளை ஒப்பிட்டு சரியான வட்டி விகிதம் தேர்ந்தெடுக்கவும்."},
    {"input": "பணவீக்கம் என்றால் என்ன?", "output": "பணவீக்கம் என்பது பொருட்கள் மற்றும் சேவைகளின் விலை காலப்போக்கில் அதிகரிப்பது. முதலீடு மூலம் இதனை சமாளிக்கலாம்."},
    {"input": "EMI கணக்கிடுவது எப்படி?", "output": "EMI = [P x R x (1+R)^N] / [(1+R)^N - 1]. இங்கு P அசல், R மாத வட்டி விகிதம், N மாதங்கள். ஆன்லைன் கால்குலேட்டர் பயன்படுத்தலாம்."},
]

_DOMAIN_SAMPLES = {
    "client_1": _HEALTHCARE_SAMPLES,
    "client_2": _EDUCATION_SAMPLES,
    "client_3": _FINANCIAL_SAMPLES,
}


def generate_client_dataset(client_id: str, output_dir: str = "storage/datasets") -> str:
    if client_id not in _DOMAIN_SAMPLES:
        raise DataGenerationError(f"Unknown client_id: {client_id}")

    profile = next(p for p in CLIENT_PROFILES if p["client_id"] == client_id)
    samples = _DOMAIN_SAMPLES[client_id]

    dataset = {
        "client_id": client_id,
        "domain": profile["domain"],
        "language": profile["language"],
        "description": profile["description"],
        "num_samples": len(samples),
        "samples": samples,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{client_id}_dataset.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return output_path


def get_all_clients_data() -> list[dict]:
    result = []
    for profile in CLIENT_PROFILES:
        client_id = profile["client_id"]
        samples = _DOMAIN_SAMPLES[client_id]
        result.append({
            "client_id": client_id,
            "domain": profile["domain"],
            "language": profile["language"],
            "description": profile["description"],
            "num_samples": len(samples),
            "samples": samples,
        })
    return result


def get_client_profile(client_id: str) -> dict:
    for profile in CLIENT_PROFILES:
        if profile["client_id"] == client_id:
            return profile
    raise DataGenerationError(f"Unknown client_id: {client_id}")
