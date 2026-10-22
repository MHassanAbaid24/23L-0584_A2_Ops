# Dataset audit summary

## Counts per class

| class | files | readable | corrupt | duplicate_files |
|---|---|---|---|---|
| ai | 539 | 539 | 0 | 2 |
| real | 436 | 436 | 0 | 0 |
| TOTAL | 975 | 975 | 0 | 2 |

## Min / median / max (readable images)

| class | metric | min | median | max |
|---|---|---|---|---|
| all | width | 183 | 1024 | 8495 |
| all | height | 135 | 768 | 7091 |
| all | file_size | 11311 | 169093 | 10195352 |
| ai | width | 225 | 1024 | 6000 |
| ai | height | 135 | 800 | 4001 |
| ai | file_size | 11738 | 199760 | 5626183 |
| real | width | 183 | 920 | 8495 |
| real | height | 158 | 720 | 7091 |
| real | file_size | 11311 | 144840 | 10195352 |

## Formats and colour modes

| class | format | files |
|---|---|---|
| ai | JPEG | 429 |
| ai | PNG | 110 |
| real | JPEG | 403 |
| real | PNG | 33 |

| class | mode | files |
|---|---|---|
| ai | L | 1 |
| ai | P | 14 |
| ai | RGB | 489 |
| ai | RGBA | 35 |
| real | L | 6 |
| real | P | 9 |
| real | RGB | 407 |
| real | RGBA | 14 |

| class | extension | files |
|---|---|---|
| ai | jpeg | 36 |
| ai | jpg | 385 |
| ai | png | 115 |
| ai | webp | 3 |
| real | (none) | 1 |
| real | gif | 1 |
| real | jpeg | 21 |
| real | jpg | 378 |
| real | png | 35 |

Animated / multi-frame files: 0

Files whose extension does not match the decoded format: 20 (so decode by content, never by extension)

| path | class | extension | format |
|---|---|---|---|
| AiArtData/.amazonaws.com2Fpublic2Fimages2Ff89de3bb-279b-42c9-9822-a653fb5b21e6_600x480.png | ai | png | JPEG |
| AiArtData/100960167.jpg | ai | jpg | PNG |
| AiArtData/1235748-ai-animals.png | ai | png | JPEG |
| AiArtData/1687776010494_Screenshot20230626160839.png | ai | png | JPEG |
| AiArtData/Earth-Species-project-generative-ai-unders.max-2100x2100.jpg | ai | jpg | PNG |
| AiArtData/Instagram_Viral_Image_AI_Cat_Dog_1680332724335_1680332736162_1680332736162.png | ai | png | JPEG |
| AiArtData/ai-generated-8146218_1280.png | ai | png | JPEG |
| AiArtData/create-text-prompts-for-you-to-create-ai-generated-arts.png | ai | png | JPEG |
| AiArtData/lake-with-mountains-digital-download-ai-art-landscape-627.webp | ai | webp | JPEG |
| AiArtData/mazonaws.com2Fpublic2Fimages2F322b8da8-dc7a-44de-b28f-c8019d88e9f7_1024x1024.png | ai | png | JPEG |
| AiArtData/new-project-1_tecm.1248.jpg | ai | jpg | PNG |
| AiArtData/psychedelic-revelation-ai-artwork-718_300x2x.webp | ai | webp | JPEG |
| AiArtData/scenery-art-landscape-ai-987.webp | ai | webp | JPEG |
| AiArtData/thejohnechen-a-1691152122-png-selar.co-20230804122842.png | ai | png | JPEG |
| RealArt/.amazonaws.com2Fpublic2Fimages2Fcb32a00a-bf52-48fe-9ba6-4e21cf4c1c57_800x800.png | real | png | JPEG |
| RealArt/1707541234037 | real | (none) | JPEG |
| RealArt/2g24m0k80001000.png | real | png | JPEG |
| RealArt/black-american-portraits-130.gif | real | gif | JPEG |
| RealArt/cute-baby-animals-at-australia-zoo.thumb.800.480.png | real | png | JPEG |
| RealArt/images37.jpg | real | jpg | PNG |

## Corrupt / unreadable files: 0

## Exact duplicates: 1 groups, 2 files, 1 redundant copies, 0 groups spanning both classes

| dup_group | path | class | file_size |
|---|---|---|---|
| 1 | AiArtData/image33.jpg | ai | 163866 |
| 1 | AiArtData/image34.jpg | ai | 163866 |
