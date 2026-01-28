# Workbook Blank Detection Prompt

Use this prompt to instruct an AI model to return **anchor** and **target** boxes for all writable blanks on a workbook page.

---

## Prompt (copy/paste)

**Task:** Identify all user‑fillable blanks (underlines, dotted/dashed lines, boxes, or empty writable spaces) in the document. Return their bounding boxes in strict JSON format.

For each fillable blank, identify **two regions**:
- **Anchor:** The bounding box of the specific word or symbol immediately to the left of the blank (e.g., “Name:”, “1.”, “a)”).
- **Target:** The bounding box of the writable blank area (line/box/space for the answer).

**Coordinate system**
- Use `[ymin, xmin, ymax, xmax]` relative to the image dimensions.
- Scale: **0–1000** (integer values).

**Rules**
- The `anchor_box` and `target_box` must refer to the same blank.
- `anchor_box` must tightly bound only the anchor text/symbol (not the blank).
- `target_box` must tightly bound only the writable blank area (not surrounding text).
- The bottom edge (`ymax`) of the `target_box` should align with the writing baseline on that line.
- Use tight heights that match the cap height of surrounding text.
- Include `filled: true` when the blank already contains handwriting or typed text.
- If no blanks are found, return `{"items": []}`.

**Workbook‑specific requirements (very important)**
- Do **NOT** return boxes for normal spaces at the end of a sentence line (right‑margin whitespace).
- For sentence‑level answer lines (e.g., “Correct these sentences …”), return **ONE** `target_box` per numbered sentence covering the full answer line(s) below it. If there are multiple consecutive answer lines for the same number, **merge them into a single** `target_box` spanning all lines.
- When a blank line has **no printed text** immediately to its left, create the `anchor_box` as a small **letter‑sized** box at the **left edge** of that blank line (same line as the blank), with its bottom aligned to the writing baseline.
- If there are several **consecutive unlabeled** blank lines (one under another), return coordinates **only for the topmost** one.

---

## JSON Schema (Single Page)

```json
{
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "anchor_box": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 4,
            "maxItems": 4
          },
          "target_box": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 4,
            "maxItems": 4
          },
          "label": { "type": "string" },
          "filled": { "type": "boolean" }
        },
        "required": ["target_box"]
      }
    }
  },
  "required": ["items"]
}
```

**Single‑page example output**
```json
{
  "items": [
    {
      "anchor_box": [140, 75, 170, 110],
      "target_box": [140, 120, 170, 320],
      "label": "",
      "filled": false
    }
  ]
}
```

---

## JSON Schema (Multiple Pages)

```json
{
  "type": "object",
  "properties": {
    "pages": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "page": { "type": "integer" },
          "items": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "anchor_box": {
                  "type": "array",
                  "items": { "type": "number" },
                  "minItems": 4,
                  "maxItems": 4
                },
                "target_box": {
                  "type": "array",
                  "items": { "type": "number" },
                  "minItems": 4,
                  "maxItems": 4
                },
                "label": { "type": "string" },
                "filled": { "type": "boolean" }
              },
              "required": ["target_box"]
            }
          }
        },
        "required": ["page", "items"]
      }
    }
  },
  "required": ["pages"]
}
```

**Multi‑page example output**
```json
{
  "pages": [
    {
      "page": 1,
      "items": [
        {
          "anchor_box": [140, 75, 170, 110],
          "target_box": [140, 120, 170, 320],
          "label": "",
          "filled": false
        }
      ]
    },
    {
      "page": 2,
      "items": []
    }
  ]
}
```
