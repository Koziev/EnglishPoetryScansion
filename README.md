# English Poetry Scansion Tool

A Python library for analyzing English syllabo-tonic poetry. It detects meter,
rhyme schemes, and stress patterns in verse.

## How It Works

The tool uses an orthoepic dictionary to determine word stresses and phonemic
forms, combined with heuristic rules for accentuation and syllabification. It
tests five common two- and three-syllable meters:
- Iamb
- Trochee
- Dactyl
- Amphibrach
- Anapest

The selected meter is the one with the fewest mismatches against the
dictionary's stress patterns.


## Output

The analysis returns:
1. **Meter**
2. **Rhyme scheme** (ABAB, AABB, etc.)
3. **Stress positions**
4. **Technicality score** (0-1), quantifying how well the poem conforms to its
meter and rhyme structure

The technicality score is particularly useful for evaluating AI-generated
poetry.


## Usage

*[Usage details to be added]*


### Example

Input (first quatrain of "The Cow" by R.L.Stiwenson):

```
The friendly cow all red and white,
I love with all my heart:
She gives me cream with all her might,
To eat with apple-tart.
```

The scansion result:

```
The fr+iendly c+ow all r+ed and wh+ite,
I l+ove with +all my h+eart:
She g+ives me cr+eam with +all her m+ight,
To +eat with +apple - t+art.
```

The stressed syllables are marked by prepending "+" sign.


