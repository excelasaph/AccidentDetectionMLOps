// MongoDB Schema Validation for Accident Detection Collections
// Run these commands in MongoDB shell or MongoDB Compass

// Schema for train collection
db.createCollection("train", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      title: "Accident Detection Training Sequences",
      required: ["sequence_id", "image_paths", "label", "split"],
      properties: {
        sequence_id: {
          bsonType: "string",
          description: "Unique identifier for the sequence"
        },
        image_paths: {
          bsonType: "array",
          description: "Array of image file paths",
          items: {
            bsonType: "string"
          },
          minItems: 1,
          maxItems: 10
        },
        label: {
          bsonType: "int",
          enum: [0, 1],
          description: "Label must be 0 (Non Accident) or 1 (Accident)"
        },
        split: {
          bsonType: "string",
          enum: ["train"],
          description: "Split must be 'train' for this collection"
        }
      }
    }
  }
});

// Schema for test collection
db.createCollection("test", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      title: "Accident Detection Test Sequences",
      required: ["sequence_id", "image_paths", "label", "split"],
      properties: {
        sequence_id: {
          bsonType: "string",
          description: "Unique identifier for the sequence"
        },
        image_paths: {
          bsonType: "array",
          description: "Array of image file paths",
          items: {
            bsonType: "string"
          },
          minItems: 1,
          maxItems: 10
        },
        label: {
          bsonType: "int",
          enum: [0, 1],
          description: "Label must be 0 (Non Accident) or 1 (Accident)"
        },
        split: {
          bsonType: "string",
          enum: ["test"],
          description: "Split must be 'test' for this collection"
        }
      }
    }
  }
});

// Schema for val collection
db.createCollection("val", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      title: "Accident Detection Validation Sequences",
      required: ["sequence_id", "image_paths", "label", "split"],
      properties: {
        sequence_id: {
          bsonType: "string",
          description: "Unique identifier for the sequence"
        },
        image_paths: {
          bsonType: "array",
          description: "Array of image file paths",
          items: {
            bsonType: "string"
          },
          minItems: 1,
          maxItems: 10
        },
        label: {
          bsonType: "int",
          enum: [0, 1],
          description: "Label must be 0 (Non Accident) or 1 (Accident)"
        },
        split: {
          bsonType: "string",
          enum: ["val"],
          description: "Split must be 'val' for this collection"
        }
      }
    }
  }
});

// Alternative: If collections already exist, add validation rules
// db.runCommand({
//   collMod: "train",
//   validator: { /* schema above */ }
// });
