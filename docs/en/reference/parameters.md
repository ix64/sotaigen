# Parameter file

[中文](../../reference/parameters.md)

Contract 3: the file's structure, the parameter names and their units. The parameters' ranges and rules
are not in the contract: they improve with the shapes, so a file may become invalid in a later
version, and a reader still opens it and says which parameter fails. The golden copy is
`tests/contracts/parameters.json`; the JSON Schema ships with the package, at
`sotaigen/data/params.schema.json` (it holds the shape and the ranges, not the rules across
parameters).

## Structure

```json
{
  "format": "sotaigen/params",
  "version": 1,
  "sex": "female",
  "base": { "name": "female-long" },
  "generator": "sotaigen 0.1.0",
  "body": { "headHeight": 254.545, "torsoLength": 390.091, "...": "..." }
}
```

| Field | What it holds |
|---|---|
| `format` | always `"sotaigen/params"` |
| `version` | the file format's version, 1 now |
| `sex` | `"female"` or `"male"`: which parameters there are, and which ranges and rules hold |
| `body` | the parameters, by name |
| `base` | optional: the body the parameters started from, `{"name": …}` |
| `generator` | optional: what wrote the file, `"sotaigen <version>"` |

## Reading a file

| Case | What a reader does |
|---|---|
| `format` is wrong, or it isn't JSON | refuses |
| `version` is newer than the reader | refuses, saying so |
| no `sex`, or it isn't a string | refuses |
| a `sex` there is none of | opens it; the check says there is no such sex |
| top-level fields it doesn't know | keeps them, and writes them back unchanged |
| `body` lacks a required parameter, or has one it doesn't know | opens it; the check fails and nothing is made; a missing one is not filled in with a default |
| outside a range, or breaking a rule | opens it; the check fails; nothing is clamped |

- **`version` goes up only when the structure or a meaning changes.** Not for ranges and rules, nor
  for a new optional parameter (without it the body is the same); for removing or changing a required
  one. A reader migrates an older version to its own instead of refusing it.
- **A file holds parameters only,** not sliders or the head count and other means of editing.
- **Numbers are written as floats** (`124.0`): JSON doesn't tell 124 from 124.0.

## Units

- A `mm` parameter is in millimetres; a `ratio` is a plain number.
- An optional parameter may be left out; the body then has its default, which the JSON Schema states.
- "Measured on" says where a value is measured. The parameters measured on "an internal control" have
  no measurement: what one value makes may differ between versions.

## Female parameters

<!-- generated: parameters-female -->
| Parameter | Unit | Required | Measured on | What it is |
|---|---|---|---|---|
| `headHeight` | mm | yes | between landmarks the bones are placed from | Head height |
| `neckLength` | mm | yes | between landmarks the bones are placed from | Neck length |
| `torsoLength` | mm | yes | between landmarks the bones are placed from | Torso length |
| `crotchHeight` | mm | yes | between landmarks the bones are placed from | Crotch height |
| `kneeHeight` | mm | yes | between landmarks the bones are placed from | Knee height |
| `ankleHeight` | mm | yes | between landmarks the bones are placed from | Ankle height |
| `biacromialBreadth` | mm | yes | between landmarks the bones are placed from | Shoulder breadth |
| `acromionRise` | mm | yes | between landmarks the bones are placed from | Acromion rise |
| `waistRise` | mm | yes | between landmarks the bones are placed from | Waist height above the crotch |
| `chestBreadth` | mm | yes | the finished surface; solved for | Ribcage breadth |
| `chestDepth` | mm | yes | the finished surface; solved for | Ribcage depth |
| `waistBreadth` | mm | yes | the surface's breadth at its landmark, within 2% | Waist width |
| `waistDepth` | mm | yes | the finished surface; solved for | Waist depth, front to back at the waist landmark |
| `hipBreadth` | mm | yes | the surface's breadth at its landmark, within 2% | Hip width |
| `hipDepth` | mm | yes | the finished surface; solved for | Hip depth |
| `upperArmLength` | mm | yes | between landmarks the bones are placed from | Upper-arm length |
| `lowerArmLength` | mm | yes | between landmarks the bones are placed from | Forearm length |
| `handLength` | mm | yes | between landmarks the bones are placed from | Hand length |
| `footLength` | mm | yes | between landmarks the bones are placed from | Foot length |
| `upperArmGirth` | mm | yes | the finished surface; solved for | Upper-arm girth |
| `wristGirth` | mm | yes | the finished surface; solved for | Wrist girth |
| `neckGirth` | mm | yes | the finished surface; solved for | Neck girth |
| `upperThighGirth` | mm | yes | the finished surface; solved for | Upper-thigh girth |
| `kneeGirth` | mm | yes | the finished surface; solved for | Knee girth |
| `calfGirth` | mm | yes | the finished surface; solved for | Calf girth |
| `ankleGirth` | mm | yes | the finished surface; solved for | Ankle girth |
| `legClearance` | mm | yes | an internal control; what a value makes may change with the shape's version | Leg clearance |
| `upperTrunkLean` | mm | yes | the finished surface; solved for | Upper-body lean back |
| `pelvisLean` | mm | yes | the finished surface; solved for | Pelvic tilt |
| `headWidthRatio` | ratio | yes | a ratio of two dimensions of one part | Head width |
| `headDepth` | ratio | yes | an internal control; what a value makes may change with the shape's version | Head depth |
| `jawBreadthRatio` | ratio | yes | a ratio of two dimensions of one part | Jaw |
| `definition` | ratio | yes | an internal control; what a value makes may change with the shape's version | Definition of the muscle masses |
| `softness` | ratio | yes | an internal control; what a value makes may change with the shape's version | Softness of the blends between masses |
| `outerThighArc` | mm | no | the surface's offset from a chord between two stations | Thigh: Outer arc |
| `abdomenFrontArc` | mm | no | the surface's offset from a chord between two stations | Belly and back: Belly arc |
| `lumbarHollow` | mm | no | the surface's offset from a chord between two stations | Belly and back: Lumbar hollow |
| `waistContourShift` | mm | no | an internal control; what a value makes may change with the shape's version | Belly and back: Waist transition height |
| `shoulderSetback` | mm | no | how far the surface moves against the same body without it | Shoulder setback |
| `thighFrontShift` | mm | no | how far the surface moves against the same body without it | Thigh: Front of the thigh |
| `gluteProjection` | mm | no | how far the surface moves against the same body without it | Glutes: Projection |
| `glutePeakShift` | mm | no | how far the surface moves against the same body without it | Glutes: Peak height |
| `gluteUpperFullness` | mm | no | how far the surface moves against the same body without it | Glutes: Upper fullness |
| `gluteLowerFullness` | mm | no | how far the surface moves against the same body without it | Glutes: Lower fullness |
| `outerHipShift` | mm | no | how far the surface moves against the same body without it | Glutes: Outer hip height |
| `gluteFoldLength` | mm | no | a dimension of the shape it adds | Glutes: Gluteal fold length |
| `gluteFoldDepth` | mm | no | how far the surface moves against the same body without it | Glutes: Gluteal fold depth |
| `groinDepth` | mm | no | how far the surface moves against the same body without it | Groin: Groin depth |
| `groinRelief` | mm | no | how far the surface moves against the same body without it | Groin: Relief around it |
| `pectoralThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Pectorals |
| `deltoidThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Deltoids |
| `trapeziusThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Trapezius |
| `latissimusThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Latissimus |
| `breastProjection` | mm | no | the most the tissue stands off the surface under it | Breasts: Projection |
| `breastBaseBreadth` | mm | no | a dimension of the shape it adds | Breasts: Base width |
| `breastFoldDrop` | mm | no | a dimension of the shape it adds | Breasts: Fold height |
| `breastUpperPoleRatio` | ratio | no | a ratio of two dimensions of one part | Breasts: Upper fullness |
| `breastGap` | mm | no | a dimension of the shape it adds | Breasts: Gap |
| `breastApexRatio` | ratio | no | a ratio of two dimensions of one part | Breasts: Front point outward |
<!-- end generated -->

## Male parameters

A male body has no breast parameters; the rest are the female ones.

<!-- generated: parameters-male -->
| Parameter | Unit | Required | Measured on | What it is |
|---|---|---|---|---|
| `headHeight` | mm | yes | between landmarks the bones are placed from | Head height |
| `neckLength` | mm | yes | between landmarks the bones are placed from | Neck length |
| `torsoLength` | mm | yes | between landmarks the bones are placed from | Torso length |
| `crotchHeight` | mm | yes | between landmarks the bones are placed from | Crotch height |
| `kneeHeight` | mm | yes | between landmarks the bones are placed from | Knee height |
| `ankleHeight` | mm | yes | between landmarks the bones are placed from | Ankle height |
| `biacromialBreadth` | mm | yes | between landmarks the bones are placed from | Shoulder breadth |
| `acromionRise` | mm | yes | between landmarks the bones are placed from | Acromion rise |
| `waistRise` | mm | yes | between landmarks the bones are placed from | Waist height above the crotch |
| `chestBreadth` | mm | yes | the finished surface; solved for | Ribcage breadth |
| `chestDepth` | mm | yes | the finished surface; solved for | Ribcage depth |
| `waistBreadth` | mm | yes | the surface's breadth at its landmark, within 2% | Waist width |
| `waistDepth` | mm | yes | the finished surface; solved for | Waist depth, front to back at the waist landmark |
| `hipBreadth` | mm | yes | the surface's breadth at its landmark, within 2% | Hip width |
| `hipDepth` | mm | yes | the finished surface; solved for | Hip depth |
| `upperArmLength` | mm | yes | between landmarks the bones are placed from | Upper-arm length |
| `lowerArmLength` | mm | yes | between landmarks the bones are placed from | Forearm length |
| `handLength` | mm | yes | between landmarks the bones are placed from | Hand length |
| `footLength` | mm | yes | between landmarks the bones are placed from | Foot length |
| `upperArmGirth` | mm | yes | the finished surface; solved for | Upper-arm girth |
| `wristGirth` | mm | yes | the finished surface; solved for | Wrist girth |
| `neckGirth` | mm | yes | the finished surface; solved for | Neck girth |
| `upperThighGirth` | mm | yes | the finished surface; solved for | Upper-thigh girth |
| `kneeGirth` | mm | yes | the finished surface; solved for | Knee girth |
| `calfGirth` | mm | yes | the finished surface; solved for | Calf girth |
| `ankleGirth` | mm | yes | the finished surface; solved for | Ankle girth |
| `legClearance` | mm | yes | an internal control; what a value makes may change with the shape's version | Leg clearance |
| `upperTrunkLean` | mm | yes | the finished surface; solved for | Upper-body lean back |
| `pelvisLean` | mm | yes | the finished surface; solved for | Pelvic tilt |
| `headWidthRatio` | ratio | yes | a ratio of two dimensions of one part | Head width |
| `headDepth` | ratio | yes | an internal control; what a value makes may change with the shape's version | Head depth |
| `jawBreadthRatio` | ratio | yes | a ratio of two dimensions of one part | Jaw |
| `definition` | ratio | yes | an internal control; what a value makes may change with the shape's version | Definition of the muscle masses |
| `softness` | ratio | yes | an internal control; what a value makes may change with the shape's version | Softness of the blends between masses |
| `outerThighArc` | mm | no | the surface's offset from a chord between two stations | Thigh: Outer arc |
| `abdomenFrontArc` | mm | no | the surface's offset from a chord between two stations | Belly and back: Belly arc |
| `lumbarHollow` | mm | no | the surface's offset from a chord between two stations | Belly and back: Lumbar hollow |
| `waistContourShift` | mm | no | an internal control; what a value makes may change with the shape's version | Belly and back: Waist transition height |
| `shoulderSetback` | mm | no | how far the surface moves against the same body without it | Shoulder setback |
| `thighFrontShift` | mm | no | how far the surface moves against the same body without it | Thigh: Front of the thigh |
| `gluteProjection` | mm | no | how far the surface moves against the same body without it | Glutes: Projection |
| `glutePeakShift` | mm | no | how far the surface moves against the same body without it | Glutes: Peak height |
| `gluteUpperFullness` | mm | no | how far the surface moves against the same body without it | Glutes: Upper fullness |
| `gluteLowerFullness` | mm | no | how far the surface moves against the same body without it | Glutes: Lower fullness |
| `outerHipShift` | mm | no | how far the surface moves against the same body without it | Glutes: Outer hip height |
| `gluteFoldLength` | mm | no | a dimension of the shape it adds | Glutes: Gluteal fold length |
| `gluteFoldDepth` | mm | no | how far the surface moves against the same body without it | Glutes: Gluteal fold depth |
| `groinDepth` | mm | no | how far the surface moves against the same body without it | Groin: Groin depth |
| `groinRelief` | mm | no | how far the surface moves against the same body without it | Groin: Relief around it |
| `pectoralThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Pectorals |
| `deltoidThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Deltoids |
| `trapeziusThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Trapezius |
| `latissimusThickness` | mm | no | an internal control; what a value makes may change with the shape's version | Latissimus |
<!-- end generated -->
