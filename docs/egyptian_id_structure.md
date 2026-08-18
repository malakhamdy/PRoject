# Egyptian National ID Card Structure Reference

## Overview
This document describes the visual structure and layout of the Egyptian National ID card based on analysis of reference samples.

**IMPORTANT**: This is for STRUCTURAL REFERENCE ONLY. Do NOT use exact pixel coordinates in production code. The system must dynamically detect fields in each image.

---

## Physical Properties

### Card Dimensions
- Standard ID-1 format: 85.60 × 53.98 mm (3.370 × 2.125 in)
- Aspect ratio: ~1.587:1
- Plastic card with rounded corners

### Orientation
- Arabic text reads RIGHT-TO-LEFT (RTL)
- Portrait photo typically on the RIGHT side
- Fields arranged in logical RTL groups

---

## Visual Layout Structure

### Primary Regions (RTL Order)

#### 1. Header Region (Top)
- Republic of Egypt branding
- May contain national emblem
- Typically spans full width

#### 2. Photo Region (Right Side)
- Portrait photograph
- Rectangular area approximately 15-20% of card width
- Located on the RIGHT (RTL layout anchor)
- Distinct visual properties (different background texture)
- Used as structural anchor for field positioning

#### 3. Personal Data Fields (Left-Center)
Arranged vertically in RTL order:

**Field 1: National ID Number (الرقم القومي)**
- 14-digit numeric sequence
- Located below header, left of photo
- High digit density region
- Horizontal text line

**Field 2: Full Name (الاسم)**
- Arabic text, typically 2-4 words
- Below NID field
- Moderate-to-high Arabic text density
- May span multiple lines

**Field 3: Date of Birth (تاريخ الميلاد)**
- Date format (DD/MM/YYYY or similar)
- Mix of Arabic/Western digits
- Compact horizontal region
- Left side, middle vertical position

**Field 4: Gender (النوع)**
- Short categorical text (ذكر/أنثى)
- Compact region
- Right side, middle vertical position (RTL pairing with DOB)

**Field 5: Governorate (المحافظة)**
- Arabic location name
- Below DOB/Gender row
- Short-to-medium Arabic text

**Field 6: Address (العنوان)**
- Multi-word Arabic text
- May span multiple lines
- Largest text block
- Bottom portion of data area

---

## Arabic Field Labels

Expected Arabic labels (may vary by ID generation):

| Field | Arabic Label | Unicode |
|-------|-------------|---------|
| National ID | الرقم القومي | \u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a |
| Name | الاسم | \u0627\u0644\u0627\u0633\u0645 |
| Date of Birth | تاريخ الميلاد | \u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f |
| Gender | النوع | \u0627\u0644\u0646\u0648\u0639 |
| Governorate | المحافظة | \u0627\u0644\u0645\u062d\u0627\u0641\u0638\u0629 |
| Address | العنوان | \u0627\u0644\u0639\u0646\u0648\u0627\u0646 |

These labels serve as visual anchors for field localization.

---

## NID Structure (14 Digits)

```
Position 1: Century code (2=1900s, 3=2000s)
Positions 2-7: Birth date (YYMMDD)
Positions 8-9: Governorate code (01-27)
Position 10: Gender indicator (odd=male, even=female)
Positions 11-13: Sequential number
Position 14: Check digit
```

Example: `2 950115 12 3 456 7`
- Century: 2 (1900s)
- DOB: 15 January 1995
- Governorate: 12 (Giza)
- Gender: 3 (odd = male)
- Sequence: 456
- Check: 7

---

## Governorate Codes

| Code | Arabic Name | English Name |
|------|------------|--------------|
| 01 | القاهرة | Cairo |
| 02 | الإسكندرية | Alexandria |
| 03 | بورسعيد | Port Said |
| 04 | السويس | Suez |
| 05 | دمياط | Damietta |
| 06 | الدقهلية | Dakahlia |
| 07 | الشرقية | Sharqia |
| 08 | القليوبية | Qalyubia |
| 09 | كفر الشيخ | Kafr El Sheikh |
| 10 | الغربية | Gharbia |
| 11 | المنوفية | Menofia |
| 12 | البحيرة | Beheira |
| 13 | الإسماعيلية | Ismailia |
| 14 | الدقهلية | Dakahlia |
| 15 | بني سويف | Beni Suef |
| 16 | الفيوم | Fayoum |
| 17 | أسيوط | Asyut |
| 18 | سوهاج | Sohag |
| 19 | قنا | Qena |
| 20 | أسوان | Aswan |
| 21 | الأقصر | Luxor |
| 22 | البحر الأحمر | Red Sea |
| 23 | الوادي الجديد | New Valley |
| 24 | مطروح | Matrouh |
| 25 | شمال سيناء | North Sinai |
| 26 | جنوب سيناء | South Sinai |
| 27 | خارج الجمهورية | Outside Republic |

---

## Visual Characteristics

### Text Properties
- Arabic script with connected characters
- Right-to-left reading order
- Mixed Arabic-Indic and Western digits possible
- Official fonts are clear, sans-serif
- Text is printed (not handwritten)

### Background
- Security patterns may be present
- Holographic elements possible
- Varies by ID generation/year
- Should be distinguished from foreground text

### Photo Region
- Rectangular portrait area
- Different background texture/color
- No text overlay typically
- Strong visual contrast with text regions

---

## Localization Strategy Implications

### Use Photo as Anchor
1. Detect photo region first (distinct visual properties)
2. Establish RTL orientation from photo position
3. Position text fields relative to photo (left of photo)

### Use Arabic Labels as Anchors
1. Search for expected Arabic label patterns
2. Field data appears near corresponding label
3. Labels provide semantic confirmation

### Content-Type Detection
1. NID: High digit density, 14-digit sequence potential
2. Name: Long Arabic text, moderate density
3. DOB: Short digit sequence with separators
4. Gender: Very short categorical text
5. Governorate: Medium Arabic location text
6. Address: Large multi-line text block

### Hierarchical Refinement
1. Start with normalized region proposals
2. Refine using photo anchor
3. Refine using label detection
4. Refine using content-type detection
5. Validate crop quality
6. Generate alternatives if uncertain

---

## Variation Handling

The system must handle:
- Different ID generations (layout variations)
- Different resolutions and scales
- Rotation (0°, 90°, 180°, 270°, arbitrary angles)
- Perspective distortion
- Lighting variations
- Partial occlusion
- Wear and tear
- Different camera distances

**CRITICAL**: Never assume fixed pixel coordinates. Always detect dynamically.

---

## Test Fixture Requirements

Synthetic test fixtures should include:
- Fake Arabic text (non-sensitive placeholder)
- Fake 14-digit NID sequences
- Photo placeholder region
- Arabic field labels
- Realistic visual complexity (>5000 unique values)
- Multiple variants (rotated, scaled, perspective)
- Varied backgrounds

---

*Document Version: 1.0*
*Last Updated: 2024*
*Purpose: Structural reference for OCI localization development*
