# SNN Workflow Web Project v2

เว็บระบบ Siamese-style image similarity สำหรับเปรียบเทียบรูปภาพ โดยเวอร์ชันนี้รัน embedding จากภาพจริงใน browser:

- ใส่ Input Dataset เองได้
- อัปโหลดรูปภาพเข้าคลาสเองได้ เช่น A, B, C หรือชื่อแมวจริง
- Pair Generator เลือกคู่ภาพจาก Input ที่อัปโหลดจริง
- สุ่ม Positive Pair จากรูปคลาสเดียวกัน
- สุ่ม Negative Pair จากรูปคนละคลาส
- แสดงภาพจริงใน Animation Flow
- ปรับ Threshold ได้
- คำนวณ Similarity Score จาก embedding ของภาพจริง ไม่ได้สุ่มคะแนน
- ใช้ MobileNet ผ่าน TensorFlow.js เป็น default encoder เมื่อโหลด CDN ได้
- มี fallback แบบ canvas embedding หากออฟไลน์หรือโหลดโมเดลไม่ได้
- มี Model Manager สำหรับโหลด TensorFlow.js model ที่ผู้ใช้อัปโหลดจริง

## วิธีรัน

แตก zip แล้วเปิด Terminal ในโฟลเดอร์นี้ จากนั้นรัน:

```bash
python3 -m http.server 5500
```

แล้วเปิด:

```text
http://localhost:5500
```

ถ้าต้องการใช้โมเดล `.h5` หรือ `.keras` ให้รัน backend แทน:

```bash
python3 server.py
```

แล้วเปิด:

```text
http://127.0.0.1:5500
```

การเปิด `index.html` ผ่าน Live Server หรือ `python3 -m http.server` ใช้ได้เฉพาะโหมด frontend/TensorFlow.js เท่านั้น และจะรัน `.h5` ไม่ได้

## Deploy

Vercel ใช้สำหรับ deploy frontend/static web ได้ แต่ไม่เหมาะกับ backend TensorFlow `.h5` เพราะ dependency ของ TensorFlow มีขนาดใหญ่เกิน Vercel Serverless Function limit

- Deploy บน Vercel: ใช้ส่วน frontend, dataset UI, TensorFlow.js model, MobileNet fallback
- ใช้ `.h5/.keras`: รัน `server.py` บนเครื่อง/server ที่ติดตั้ง TensorFlow ได้ เช่น Render, Railway, VM หรือ local server

## หมายเหตุ

คะแนน similarity ตอนนี้คำนวณจากภาพจริงแล้ว ภาพสองฝั่งผ่าน encoder เดียวกันแล้วคำนวณ cosine similarity ตามแนว Siamese Network

default encoder คือ MobileNet ที่โหลดจาก CDN:

- `@tensorflow/tfjs`
- `@tensorflow-models/mobilenet`

ถ้าเครื่องไม่มีอินเทอร์เน็ต ระบบจะ fallback ไปใช้ canvas embedding จากพิกเซลจริงแทน เพื่อให้เว็บยังใช้งานได้

## การใช้โมเดล SNN ที่อัปโหลด

Model Manager ใช้ได้สองแบบ:

- ถ้าโมเดลมี 1 input ระบบถือว่าเป็น encoder: ส่งรูปทีละฝั่งเข้าโมเดล แล้วคำนวณ cosine similarity จาก embedding
- ถ้าโมเดลมี 2 input ระบบถือว่าเป็น SNN pair model: ส่งรูปซ้ายและขวาเข้าโมเดลโดยตรง แล้วใช้ output แรกเป็นผลลัพธ์ของโมเดล
- ระบบ infer ชนิด output จากโมเดลอัตโนมัติ: sigmoid/softmax จะแสดงเป็น `Similarity Score`; distance/lambda/linear pair output จะแสดงเป็น `Distance`
- ถ้า output เป็น `Distance` ระบบจะแปลงเป็น similarity ด้วย `1 / (1 + distance)` ภายใน เพื่อให้ threshold ยังใช้ตรรกะเดิมคือยิ่งมากยิ่งเหมือน
- ไฟล์โมเดลที่อัปโหลดจะถูกเก็บใน IndexedDB ของ browser ทำให้กดสลับกลับไปใช้โมเดลในประวัติได้
- TensorFlow.js `model.json` + `weights.bin` รันใน browser
- Keras `.h5` / `.keras` รันผ่าน backend `server.py` และเก็บไฟล์ไว้ในโฟลเดอร์ `server_models/`

ข้อจำกัด: `.pt` และ `.onnx` ยังไม่รองรับ ต้องแปลงเป็น TensorFlow.js หรือ Keras ก่อน
# snn_workflow_web_project_v2
