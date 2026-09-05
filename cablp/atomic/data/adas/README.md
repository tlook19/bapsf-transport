# ADAS data for helium, carbon, oxygen, boron, chromium, iron, nickel, molybdenum, and tungsten

**The data files themselves are NOT in this repository — fetch them by hand
before running anything that uses them.** OPEN-ADAS distributes its own data
and its terms forbid redistribution and publication on a public website, so
this public repository tracks only this README; the `*.dat` files are
untracked and `.gitignore`d. This directory is where they must land.

Two ADAS data formats live here: adf11 iso-nuclear master files (the bulk of
the directory, consumed by the transport model) and, for helium only, two
adf15 photon-emissivity-coefficient files (diagnostic-side, see the adf15
section below).

## Fetching the files

Download each file from the URL in the table below and save it into this
directory (`cablp/atomic/data/adas/`) under the **local filename** given in
the first column. Nothing else is needed — there is deliberately no fetch
script, and the loader reads the files straight from this directory.

Two names differ from their URL basename: OPEN-ADAS serves the adf15 files
under names containing `#`, which is awkward in shell and Python paths, so
they are stored here with `#` replaced by `_`. The `][` in the adf15 URLs is
OPEN-ADAS's own escaping of `#` in its download paths, so `curl` needs
`-g`/`--globoff` to fetch them literally:

```bash
curl -o scd96_he.dat 'https://open.adas.ac.uk/download/adf11/scd96/scd96_he.dat'
curl -g -o pec96_he_pju_he0.dat 'https://open.adas.ac.uk/download/adf15/pec96][he/pec96][he_pju][he0.dat'
```

The helium files are the only ones the transport model itself needs
(`scd96_he.dat`, `acd96_he.dat`, `plt96_he.dat`, `prb96_he.dat`); the rest
are needed only by the two analysis scripts named in the sections below.
A file that is missing when a loader reaches for it raises a `RuntimeError`
naming the file and pointing back at this README.

## Checksums, and what a mismatch means

**Verify the data-block checksum, not the whole file.** OPEN-ADAS stamps
every download with its own retrieval date, in a `C on DD-Mon-YYYY.` line
inside the file's trailing comment block, so two downloads of the *same*
ADAS revision on different days differ by that one line and therefore have
different whole-file checksums. Both parsers here (`read_adf11` in
`cablp.atomic.adas`, `read_adf15` in `scripts/score/pec_band_fractions.py`) stop
at the first line beginning with `C`, so the comment block — the date stamp
included — never reaches the numerics.

| check | command | meaning |
|---|---|---|
| **data block** (use this) | `sed -n '/^C/q;p' FILE \| sha256sum` | the bytes the parsers actually consume |
| whole file | `sha256sum FILE` | matches only a byte-identical copy of the pinned file, i.e. the same revision downloaded on the same day |

**A data-block mismatch means you have a different ADAS revision from the
one this campaign is pinned to, and the golden baseline and the 4k digest
gate will fail** — the tabulated coefficients feed the solver's reaction
rates directly, so a revision change moves the trajectory. Re-check the URL
before assuming the file is corrupt; if OPEN-ADAS has genuinely rotated the
revision, that is a stance-level event, not something to work around. A
whole-file mismatch whose only difference is the `C on ...` date line is
expected and harmless (`diff` against a known-good copy will show it).

Values below were computed from the working copies of record: the helium
'96 files retrieved 2026-07-18, the remaining adf11 files 2026-07-21, the
two adf15 files 2026-08-26, and the metastable-resolved helium '96r files
2026-09-05.

| local filename | OPEN-ADAS download URL | sha256 (data block) | sha256 (whole file, as pinned) |
|---|---|---|---|
| `acd89_b.dat` | `https://open.adas.ac.uk/download/adf11/acd89/acd89_b.dat` | `7dd11544ed5d95280959e27bd35451f73fec54c67dda69d85e3e84b4a2043d77` | `c1888ece0a471752b154dccfcfa122e0e5c2e8dd8829ba2020e19b8f5345efd2` |
| `acd89_cr.dat` | `https://open.adas.ac.uk/download/adf11/acd89/acd89_cr.dat` | `6853e1da4adbbb19463f44555e9581fddbf3d61def64bae24f58e221f580f267` | `aa87bd9bb4d2b0512f0cbc9cb3211a96569ad70f414a54ed19fe08f78a46d140` |
| `acd89_fe.dat` | `https://open.adas.ac.uk/download/adf11/acd89/acd89_fe.dat` | `6dc11673fdf020a4a03f831baa04be3778fc7e1ddcfc983e9138b7fb13924559` | `d9635bf15664fd32ba5764002ba7183aa37f42156ba07ac5a33cf0c0143c3a6a` |
| `acd89_mo.dat` | `https://open.adas.ac.uk/download/adf11/acd89/acd89_mo.dat` | `589b5f5db750f7d24013793900bd04ccfb8c7e0ddaab75260b033d49ba37af34` | `00c3be7bf5848927aa30f780fd6d8c936f013a5f1eff9f2af93a1ab2edcdbb98` |
| `acd89_ni.dat` | `https://open.adas.ac.uk/download/adf11/acd89/acd89_ni.dat` | `41307c733b58d7999b922b0b52f12f678b1199b79a5b1ea0b80a1c5f84fc69d1` | `0ac89d64da08a8dca1465997b10033334ac440c99f98302471959bdfdf22ab39` |
| `acd89_w.dat` | `https://open.adas.ac.uk/download/adf11/acd89/acd89_w.dat` | `0df957b45f37cf525a0a2f90951d7e819bf422550b6d0933457a323500aa9f96` | `d56db8ab13e1d992bff8c76cae6e465dc1617c16d0ffb8fac653263a8cd87b46` |
| `acd96_c.dat` | `https://open.adas.ac.uk/download/adf11/acd96/acd96_c.dat` | `a1b225987a02a17178e030925462376fb0156c20883cf7e69c66ac40dd5c29c2` | `9d3a9c136930837de9a52068b422569cbb279aff12cd2c08899eceb589f1458d` |
| `acd96_he.dat` | `https://open.adas.ac.uk/download/adf11/acd96/acd96_he.dat` | `cf313462e295a38dacd4a17361f9d6696b5aadf49c99ef4d7752bf98b1f94105` | `7e6d133df9a6371265e89788411a5f17c75768c2f7e3ccfd8cc59eb680de890b` |
| `acd96_o.dat` | `https://open.adas.ac.uk/download/adf11/acd96/acd96_o.dat` | `f40b0e80b4f5ce5ade3752518608affe079a6388437226e47afb27c5a1ed055c` | `200e12970901fabc94638f755add58ccfff4a45059ee63ea326ae212906c7ad6` |
| `acd96r_he.dat` | `https://open.adas.ac.uk/download/adf11/acd96r/acd96r_he.dat` | `41dbaacf03ed2d34e55cba87f84b5d2698a82a357c669f641e04bb6c02cc78fa` | `0b93be70c2c0d28afc4e117507163d29ec7e0049f04278b536c6d5ca0b9eb203` |
| `pec96_he_pju_he0.dat` | `https://open.adas.ac.uk/download/adf15/pec96][he/pec96][he_pju][he0.dat` | `3764cc2496b8a630b08bfbb0abfbd44d74d164356c08afc2f3292907d36004da` | `131bd4811537ffb25b92db7bc9368f0374b352faaf276b57c2df1521546626b8` |
| `pec96_he_pju_he1.dat` | `https://open.adas.ac.uk/download/adf15/pec96][he/pec96][he_pju][he1.dat` | `3c70e3395ec86de05d230424fe2bf2c0451c66ab4cd611dd696630fba6d279e5` | `220ffcc2f69a030ddee2ebf82f14b9ada69faa53ed00ee094f2499d4d8866a82` |
| `plt89_b.dat` | `https://open.adas.ac.uk/download/adf11/plt89/plt89_b.dat` | `d1af66a5436055feab935b05b729bcd78e04a84fd3bbcb8f7c4f54ad8a4a39bd` | `89531fccb03284bfc6a3e34964c665a26190ba406d56b8e115fcadf55577ce5a` |
| `plt89_cr.dat` | `https://open.adas.ac.uk/download/adf11/plt89/plt89_cr.dat` | `abe372eadceba4346b7902ccfc4cd3ecdf079574b53a29c26e6284f559337098` | `9f2885e339c83d49fc149f4d518b3891e87928a4637d43f5fa1bfe42e4059965` |
| `plt89_fe.dat` | `https://open.adas.ac.uk/download/adf11/plt89/plt89_fe.dat` | `bc8b5c4fef0b4354c51e59842c47b444adfe9ddf9014b45b49f1a012ec766807` | `322219d799d9b2d1eee4d12f7d92f2adece113a7b63ef1b7596a4b025097fa1b` |
| `plt89_mo.dat` | `https://open.adas.ac.uk/download/adf11/plt89/plt89_mo.dat` | `7a19478bcae546e08b97740674c11cfa42eaf09c51b2dd25ea3e729a0c755c28` | `97de6a1b09d35f3172da6264ebff6d125a94fceaf1e840df227aa9500537553b` |
| `plt89_ni.dat` | `https://open.adas.ac.uk/download/adf11/plt89/plt89_ni.dat` | `3ab19a3eb97b2eff61a7f8c171bde0c5bb4aff5c8ccf652c9d6caae1ddd9ae5f` | `e236d7f7a00263d875d3980315b939ce7d81091fcadf4ae73a2d03eb5bac84d7` |
| `plt89_w.dat` | `https://open.adas.ac.uk/download/adf11/plt89/plt89_w.dat` | `ee264369a246f57993a3cdc728f3ecd0d2b0a5b2c61ed1c5c43f4beaddcce181` | `251549facb6e0214999350043ffc343b5ac57078f82331a7121756c600f4aa66` |
| `plt96_c.dat` | `https://open.adas.ac.uk/download/adf11/plt96/plt96_c.dat` | `b5754d09391257a89d048165e197c7c4c96842111c329ca67b420e022c01092d` | `4aee0080e800740ef1dd916c4550d0e9bd47c6bd26bfd19e9311b51e69adebf1` |
| `plt96_he.dat` | `https://open.adas.ac.uk/download/adf11/plt96/plt96_he.dat` | `9d61717b63c00c0cb21b9d151f2e280ff26eb1377e84dc84e562d29f97ce6585` | `bccc1e906f900401e86d1028aa40073cb703276a06fdcb957394873a285e31f1` |
| `plt96_o.dat` | `https://open.adas.ac.uk/download/adf11/plt96/plt96_o.dat` | `5ee2ae62ccb0db6cd900f647ad5ae423a29ad84f00874aed0154f97ab6ae65b8` | `7ebc07701d7b5e6c9123bc263f5bc274f00a4a923e7db55b92d7883205daecf7` |
| `plt96r_he.dat` | `https://open.adas.ac.uk/download/adf11/plt96r/plt96r_he.dat` | `023ab10023721aa1957e55abc022c8ba320416edaf35245f8444e45c791cf71a` | `acdf38f3349983f278e91eb7f5cf0e94752db0df76d24c3bf45a5ff634840188` |
| `prb89_b.dat` | `https://open.adas.ac.uk/download/adf11/prb89/prb89_b.dat` | `e00bc0019030abc9fcce1646f76d33afa491f9079f8425f36d9eba46e34ea710` | `23057309dbce0df60d61c50113b10d46007ec1afb98d9e8eaebb43e6aa9b27a2` |
| `prb89_cr.dat` | `https://open.adas.ac.uk/download/adf11/prb89/prb89_cr.dat` | `94a274f2715418cf9fa20af9e5f6c50511f696857866605f35992695d781615d` | `937042ed61b5fff03e2db34bbda5a103724663030734166f8545826ddc903073` |
| `prb89_fe.dat` | `https://open.adas.ac.uk/download/adf11/prb89/prb89_fe.dat` | `fbea8ad914894db96fba5cd12252ce1bb8697d9c62b583c9bc2bac8c2ebf69fd` | `9297181870668ab458e02727159a2d6faaecbdad9c5e9d2e8ac08e614e0a34a2` |
| `prb89_mo.dat` | `https://open.adas.ac.uk/download/adf11/prb89/prb89_mo.dat` | `fbce9231a8d74309e6170c2189e823b4a70b5d67d145e153e3de0bae13a0dc1e` | `4857bb6c61e8d18467ccf829700dc5857c54e0a5ae060ef815873dc7e8b05b5a` |
| `prb89_ni.dat` | `https://open.adas.ac.uk/download/adf11/prb89/prb89_ni.dat` | `3eac34940c85dc59a4dd50f1a0ce41d8a32b65cb039893056a01fa7ce8786f28` | `1df49a8ab995c0f02025d7a99f9aa470ccdb479c8e33269e8a81b10e1c8ce373` |
| `prb89_w.dat` | `https://open.adas.ac.uk/download/adf11/prb89/prb89_w.dat` | `ffa76d06b9a09847751ae6cb1ec1b76dbc557f70e1f2bea67c9dce8ccf5ad3f2` | `2f72fc0c0234a6322bf9a870b61d534e8abe3d3713577bcc357e40df8731087d` |
| `prb96_c.dat` | `https://open.adas.ac.uk/download/adf11/prb96/prb96_c.dat` | `2c139732a51db9c035550ebf2d832d1ba12655b5c58d04f4bc567dfd29ef6041` | `746405c8eaeb80e3aa732e2b4cfd12094d16119c5b67a6fe4febcf9c68ae1a44` |
| `prb96_he.dat` | `https://open.adas.ac.uk/download/adf11/prb96/prb96_he.dat` | `9743b4ad6effd37b35767d923aa37d1252ac8829211b0c70dd9d32336fb1222d` | `0d08e6cd76d5e9e51f82b29fc794bc2a043d3d4bf669d7acc4d67142ce03b907` |
| `prb96_o.dat` | `https://open.adas.ac.uk/download/adf11/prb96/prb96_o.dat` | `5ca57b76954c7ad7fac832e252af103e376675bc531c9fabfb102782075e2f5f` | `fc8963c8c35056cc313904835cd077f0db8a0ce48f588a7dc3334a107688fad2` |
| `prb96r_he.dat` | `https://open.adas.ac.uk/download/adf11/prb96r/prb96r_he.dat` | `7c8b77bb46a2b4a6d2d8c7bee444e0fbfffe0dde39ee79396e2497d3159c2e10` | `c9ebcf75281c6026ece4d0e17c6c064e13f034b4eb1e6249d005419810840346` |
| `qcd96r_he.dat` | `https://open.adas.ac.uk/download/adf11/qcd96r/qcd96r_he.dat` | `ef0ab1014b917b602b8fb7a80e3ed656747715c89b2cebaaf5362e2fe69e35c4` | `ca78fa6676867097d83bfa16c37885f8ad0299ef6deb83a5a641630b57e3ce25` |
| `scd89_b.dat` | `https://open.adas.ac.uk/download/adf11/scd89/scd89_b.dat` | `4d148987dda4f19cc6f6946f1fad75b5000aa7c634fdf611fba74b3255fb8192` | `6fa3b82f4d1239ac90ef650fc2ee05dfc4569d5f1eff12b5f205878fc216681e` |
| `scd89_cr.dat` | `https://open.adas.ac.uk/download/adf11/scd89/scd89_cr.dat` | `91430740e7a4e996b1fc9043da32063ef65a0a560e87be7a71c1c26791caacc6` | `15808b53142ce22b3d3d590be0c470b7bace7d7dc12f44474f0e33380dea6ab2` |
| `scd89_fe.dat` | `https://open.adas.ac.uk/download/adf11/scd89/scd89_fe.dat` | `3a74ac7cd3dc8ad7cfce7d20cac63f7bc7d84eb5e3b9f3b3d05973fe73b7546d` | `360f693f94a6608ea1196f3473e77ec644e9d7ee06b534f12fb91afa8ebde751` |
| `scd89_mo.dat` | `https://open.adas.ac.uk/download/adf11/scd89/scd89_mo.dat` | `97d680b2239f9cee6f09b21ce0ce9ef3c2c3d1a82a73f08921a5aec60306f47e` | `27afd1611fbf51db9038a4a36ed5f322fef70c4d94c4d6e3d38a2debc2a9a003` |
| `scd89_ni.dat` | `https://open.adas.ac.uk/download/adf11/scd89/scd89_ni.dat` | `f185510d284c2fc840028a4596755eb688284b5ac12d3eb0d168923793637637` | `ef46c2ebcafbd5da41630e1f0c75606493686bc0f21e7ea9ac4504dd67d78e19` |
| `scd89_w.dat` | `https://open.adas.ac.uk/download/adf11/scd89/scd89_w.dat` | `19b09914e7eca0f0e5024be63a41ff59dfe1c435a80c601b92e650a54fef6f8b` | `52059843c0c44948c55252bf8ce295183f0306dd5e9d1c87e90cdf63baab7919` |
| `scd96_c.dat` | `https://open.adas.ac.uk/download/adf11/scd96/scd96_c.dat` | `e91340d2126fecf7657c1db33a1c86f1cff9771c0ec6a8ea18238442fb0052c8` | `0a92d5c9951c4f3fa01a2c57873f9bea6c8d65336ea5472d146f928e1b8daf72` |
| `scd96_he.dat` | `https://open.adas.ac.uk/download/adf11/scd96/scd96_he.dat` | `b41d69571140649d898ded9e7784b48209de661c3a2eb590faf1fc217c6249f1` | `c3222ec8122f1986b0e13873312bdebb0598a3efec321e18e340a3dd1819759a` |
| `scd96_o.dat` | `https://open.adas.ac.uk/download/adf11/scd96/scd96_o.dat` | `efdb55f4ebde2849c2274bc08420167b74b2d6408b39105854ca31684981868d` | `2fd82bcdf1a478c2dcdc4587bb8db4505c39dde64ec85e978b04217f7b603cad` |
| `scd96r_he.dat` | `https://open.adas.ac.uk/download/adf11/scd96r/scd96r_he.dat` | `11d2f2591af7574340677ea7e39056fb55a108029b10c0baf0762ca2612353c4` | `49b8ad0d69bc2be3b192ab22302379d8fbd95fa5d1de1d0cd6db2815c01c03e6` |

## adf11 — iso-nuclear master files

Iso-nuclear master files from OPEN-ADAS (https://open.adas.ac.uk),
unresolved (stage-to-stage) form, served at
`https://open.adas.ac.uk/download/adf11/<class><yy>/<class><yy>_<element>.dat`
(the per-file URLs above follow that pattern). Helium ('96 GCR series)
retrieved 2026-07-18; carbon/oxygen ('96) and boron/chromium/iron/nickel/
molybdenum/tungsten ('89 Abels-van Maanen series — no '96 files exist for
these) retrieved 2026-07-21. The '89 series is older average-ion-era data;
treat it as order-of-magnitude at LAPD temperatures. Lanthanum (the other
LaB6 constituent) has NO adf11 data on OPEN-ADAS in any series; tungsten is
kept as the heavy-element analog for La-class radiators.

| file | class | quantity | units | normalization |
|---|---|---|---|---|
| `scd96_he.dat` | SCD | effective ionization coefficient, stage z1-1 -> z1 | cm^3 s^-1 | per n_e * n(z1-1) |
| `acd96_he.dat` | ACD | effective recombination coefficient, stage z1 -> z1-1 (radiative + dielectronic + three-body) | cm^3 s^-1 | per n_e * n(z1) |
| `plt96_he.dat` | PLT | line power driven by excitation of ions of charge z1-1 | W cm^3 | per n_e * n(z1-1) |
| `prb96_he.dat` | PRB | recombination + bremsstrahlung power of ions of charge z1 | W cm^3 | per n_e * n(z1) |

For helium (Z = 2) each file carries stages z1 = 1, 2. The transport model
uses z1 = 1 of SCD/ACD/PLT/PRB (He0 ionization, He+ recombination, He0 line
power, He+ recombination power) and z1 = 2 of PLT (He+ line power).

The carbon (`*96_c.dat`, z1 = 1-6), oxygen (`*96_o.dat`, z1 = 1-8), boron
(`*89_b.dat`, z1 = 1-5), molybdenum (`*89_mo.dat`, z1 = 1-42), and
tungsten (`*89_w.dat`, z1 = 1-74) files were
added for the impurity-radiation scoping study
(`scripts/atomic/scope_impurity_radiation.py`): equilibrium stage balance from
SCD/ACD, total radiated power L_z from PLT+PRB. No model path consumes them
as of 2026-07-21 — the scoping verdict (required n_z/n_e ~ 4-10 % at
equilibrium for every species tested >> the ppm hypothesis) stopped the
campaign before any sink term was wired in. The stainless species
(`*89_cr.dat`, `*89_fe.dat`, `*89_ni.dat`) were added in the same event as
the last untested radiators for that study's fixed-fraction hypothesis.

These are generalized collisional-radiative (GCR) coefficients: they are
tabulated on a log10(n_e) x log10(T_e) grid (24 x 30; 5e7-2e15 cm^-3,
0.2-1.5e4 eV) and include finite-density effects (stepwise ionization via
metastables, collisional de-excitation), which the coronal Janev-era fits in
`cablp.atomic.fits` / `cablp.atomic.cross_sections` do not. Selected by the sim1d
`atomic_rate_model = "adas"` input; the historical fits remain available as
`"janev"` (the default).

File format: adf11 (see `cablp.atomic.adas.read_adf11`). All tabulated
values are log10 of the coefficient in the units above.

### adf11 — metastable-RESOLVED helium ('96r), diagnostic-side

The `*96r_he.dat` files are the metastable-resolved siblings of the helium
'96 masters above, served from the `<class>96r` directories. They carry the
same producer, code (ADAS404) and 04/11/99 date as the unresolved files —
the unresolved set is ADAS404's projection of this one — and are tabulated
on the identical 24 x 30 (n_e, T_e) grid. Retrieved 2026-09-05.

Two format differences make them unreadable by `read_adf11`, and
`cablp.atomic.adas.read_adf11_resolved` exists for them: a metastable-count
line follows the header, and every data block is introduced by a header
naming its metastable indices. For helium that count line reads `2 1 1` —
**He0 carries TWO metastables, not three.** The second is 1s2s 3S: the
low-T_e slope of `ln(QCD_1->2 / QCD_2->1)` over the 1–15 eV nodes returns
19.75–19.78 eV against the 2^3S term energy 19.820 eV, where 2^1S would
require 20.616 eV. The singlet metastable 2^1S is not an independent
population in this dataset; it sits inside the collisional-radiative bundle
built on the ground state.

| file | class | blocks (z1 = 1) | quantity |
|---|---|---|---|
| `scd96r_he.dat` | SCD | `IPRT=1/IGRD=1`, `IPRT=1/IGRD=2` | ionization out of He0 metastable IGRD into He+ parent IPRT |
| `acd96r_he.dat` | ACD | `IPRT=1/IGRD=1`, `IPRT=1/IGRD=2` | recombination of He+ parent IPRT into He0 metastable IGRD |
| `qcd96r_he.dat` | QCD | `IGRD=1/JGRD=2`, `IGRD=2/JGRD=1` | He0 metastable cross-coupling; the `1/2` block is the one carrying the ~19.8 eV threshold, i.e. ground → metastable |
| `plt96r_he.dat` | PLT | `IGRD=1/IPRT=0`, `IGRD=2/IPRT=0` | line power driven by excitation of He0 metastable IGRD |
| `prb96r_he.dat` | PRB | `IPRT=1/IGRD=0` | recombination + bremsstrahlung power of He+ parent IPRT |

**`xcd96r_he.dat` does not exist on OPEN-ADAS** (the download returns "File
not found in database", checked 2026-09-05). That is correct rather than a
gap: XCD is parent cross-coupling, which needs two metastables in the
recombining stage, and He+ has one — the count line's second entry is `1`.

ADAS404 collapses this resolved set to the unresolved masters above using the
ADF10 equilibrium metastable fractions, which balance spontaneous emission
against collisional excitation and de-excitation ONLY — no ionization loss out
of the metastable, no recombination feed into it. An ionizing plasma pays that
ionization loss, so its quasi-static rate sits BELOW the unresolved table;
a plasma at full ionization balance is additionally fed by recombination and
sits above it. The unresolved table lies between the two.

No model path consumes these. They were added for
`scripts/atomic/metastable_bracket.py`, which contracts them back to a
ground-referenced effective ionization coefficient to size the metastable /
stepwise channel and gates on that ordering; that script's docstring carries
the method.

## adf15 — helium photon emissivity coefficients (PEC)

Line-resolved PECs from the OPEN-ADAS `pec96#he` series, `pju` (unresolved,
projected-to-unresolved) variant — the variant that matches the unresolved
adf11 masters above. The metastable-**resolved** `pjr` sibling is
deliberately NOT held here: it would not compose with the unresolved PLT.
Both files retrieved 2026-08-26. Their canonical OPEN-ADAS names are the
`#` forms, which are also recoverable from each file's own trailing SCCS
header line:

| local filename | canonical name |
|---|---|
| `pec96_he_pju_he0.dat` | `pec96#he_pju#he0.dat` |
| `pec96_he_pju_he1.dat` | `pec96#he_pju#he1.dat` |

Produced by population code ADAS208, producer M. O'Mullane, dated
01.11.1999. `he0` (He I) carries 15 transitions, `he1` (He II) 9, each
tabulated for both `EXCIT` (excitation-driven) and `RECOM`
(recombination-driven) emission, so 30 and 18 blocks respectively.
Wavelengths are in Angstrom, VACUUM. PEC units are photons cm^3 s^-1, to be
multiplied by n_e * n(emitting stage). Values are tabulated linearly (NOT
log10, unlike adf11) on a 24 x 24 (n_e, T_e) grid, and **the two adf15 grids
differ from each other and from the shared adf11 grid** — `he0` spans
1e1-1e15 cm^-3 and 0.0431-259 eV, `he1` spans 1.28e3-1.28e17 cm^-3 and
0.172-1030 eV — so each file must be interpolated on its own axes. A value
of 1.00E-74 is the format's zero sentinel.

No model path consumes these: they were added for the window-transmission
band-fraction analysis (`scripts/score/pec_band_fractions.py`), which splits the
adf11 PLT line power into optical bands to establish what fraction of the
He radiated power is observable through the LAPD port windows.

Citation: H.P. Summers, "The ADAS User Manual, version 2.6" (2004),
http://www.adas.ac.uk; data via OPEN-ADAS, ADAS Project.
