# MobileNet Embedding + Cosine

ไฟล์นี้คือ baseline model สำหรับเทียบความคล้ายของหน้าแมว:

- `mobilenet_pair_similarity.keras`
  - รับ input 2 รูป: `left_cat_face`, `right_cat_face`
  - ขนาดรูป: `224x224x3`
  - ค่า pixel: `0..1`
  - output: `similarity_score` ช่วง `0..1`
  - decision: `score >= 0.8` คือคลาสเดียวกัน

- `mobilenet_embedding_encoder.keras`
  - รับ input 1 รูป
  - output เป็น L2-normalized embedding
  - ใช้ในกรณีต้องการ cache embedding แล้วคำนวณ cosine เอง

หลักการ:

```text
รูปซ้าย -> MobileNetV2 ImageNet -> embedding
รูปขวา -> MobileNetV2 ImageNet -> embedding
embedding ทั้งสอง -> cosine similarity
similarity = (cosine + 1) / 2
threshold = 0.8
```

หมายเหตุ: โมเดลนี้เป็น pretrained ImageNet baseline ไม่ใช่ SNN ที่ fine-tune ด้วย cat identity pairs โดยตรง
