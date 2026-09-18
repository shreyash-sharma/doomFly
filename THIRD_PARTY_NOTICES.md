# Third-party notices

The root MIT license covers DOOMFLY's original code, original arena definitions,
tests and original documentation. It does not replace third-party licenses or
relicense datasets, rendered game artwork, research source data or trademarks.

## Game engine, artwork and model references

- **Freedoom 0.13.0:** copyright 2001–2024 Contributors to the Freedoom project.
  The simulator explicitly selects the installed `freedoom2.wad`. Retained game
  screenshots contain Freedoom artwork. Preserve the complete
  [BSD-3-Clause notice](licenses/Freedoom-BSD-3-Clause.txt) when redistributing it.
  Source: https://github.com/freedoom/freedoom/tree/v0.13.0
- **ViZDoom 1.3.0:** original ViZDoom interface code is MIT licensed. Its exact
  copyright and permission notice is in [ViZDoom-MIT.txt](licenses/ViZDoom-MIT.txt),
  extracted from the release's `include/ViZDoom.h` header.
  Source: https://github.com/Farama-Foundation/ViZDoom/tree/1.3.0
  The optional [native observer patch](deploy/doomfly/native-observer/observer.patch)
  changes ViZDoom/ZDoom renderer files. It is subject to their original per-file
  licenses, not a blanket application of the root MIT license. The build script
  fetches the pinned official source locally; compiled engines are not committed.
  The underlying ZDoom engine includes code under other licenses. The ViZDoom
  MIT notice does not cover the entire engine. This repository distributes no
  engine executable, engine source checkout or commercial Doom IWAD. Preserve
  the complete upstream notices and applicable source obligations if distributing
  an engine/package/container binary; installing a dependency is not relicensing it.
- **Shiu model reference:** copyright 2023 Philip Shiu and Nico Spiller, MIT.
  [Original notice](licenses/Shiu-model-MIT.txt). The whole-brain LIF model is a
  scientific reference; the upstream checkout and graph are not bundled.
  Source: https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960

## Copied UI components and installed viewer dependencies

`doom-ui/components/ui/` contains adapted shadcn/ui components. Preserve the
shadcn copyright and MIT permission notice; the project's root notice is not a
replacement. The following notices are copied verbatim from installed distributions or
identified upstream license files. Their versions match the viewer's package/lock files. The inventory
also records SHA-256 hashes. Dependencies themselves are installed separately.

| Package | License | Full upstream notice |
| --- | --- | --- |
| `@base-ui/react` 1.7.0 | MIT | [LICENSE](licenses/npm/base-ui__react/LICENSE) |
| `@openai/sites-vite-plugin` 0.2.0 | MIT | [LICENSE](licenses/npm/openai__sites-vite-plugin/LICENSE) |
| `@shadcn/react` 0.3.0 | MIT | [LICENSE.md](licenses/npm/shadcn__react/LICENSE.md) |
| `class-variance-authority` 0.7.1 | Apache-2.0 | [LICENSE](licenses/npm/class-variance-authority/LICENSE) |
| `clsx` 2.1.1 | MIT | [license](licenses/npm/clsx/license) |
| `cmdk` 1.1.1 | MIT | [LICENSE.md](licenses/npm/cmdk/LICENSE.md) |
| `date-fns` 4.1.0 | MIT | [LICENSE.md](licenses/npm/date-fns/LICENSE.md) |
| `lucide-react` 1.31.0 | ISC | [LICENSE](licenses/npm/lucide-react/LICENSE) |
| `react` 19.2.6 | MIT | [LICENSE](licenses/npm/react/LICENSE) |
| `react-day-picker` 9.8.1 | MIT | [LICENSE](licenses/npm/react-day-picker/LICENSE) |
| `react-dom` 19.2.6 | MIT | [LICENSE](licenses/npm/react-dom/LICENSE) |
| `react-resizable-panels` 4.5.8 | MIT | [LICENSE.md](licenses/npm/react-resizable-panels/LICENSE.md) |
| `react-server-dom-webpack` 19.2.6 | MIT | [LICENSE](licenses/npm/react-server-dom-webpack/LICENSE) |
| `recharts` 3.8.0 | MIT | [LICENSE](licenses/npm/recharts/LICENSE) |
| `shadcn` 4.18.0 | MIT | [LICENSE.md](licenses/npm/shadcn/LICENSE.md) |
| `tailwind-merge` 3.6.0 | MIT | [LICENSE.md](licenses/npm/tailwind-merge/LICENSE.md) |
| `tw-animate-css` 1.4.0 | MIT | [LICENSE](licenses/npm/tw-animate-css/LICENSE) |
| `vinext` 1.0.0-beta.5 | MIT | [LICENSE](licenses/npm/vinext/LICENSE) |
| `embla-carousel-react` 8.5.2 | MIT | [LICENSE](licenses/npm/embla-carousel-react/LICENSE) |
| `input-otp` 1.4.2 | MIT | [LICENSE](licenses/npm/input-otp/LICENSE) |

The table inventories direct viewer dependencies and the Sites build plugin;
it is not an assertion that every transitive package or ZDoom component is MIT.
Retain the full notices supplied with any dependency you redistribute.

## Data and scientific attribution

MaleCNS v1.0 and the identified Huang et al. source-data extracts retain
[CC BY 4.0](licenses/CC-BY-4.0.txt). See [THIRD_PARTY.md](THIRD_PARTY.md) for
creators, source links, modifications and the exact material covered.
DOOMFLY's MIT license does not replace those attribution requirements.

## Independent project

DOOMFLY is an independent research experiment. It is not affiliated with,
sponsored by or endorsed by id Software, Bethesda, ZeniMax, the Freedoom project,
Farama Foundation, the dataset creators or the cited researchers. DOOM and
related marks belong to their respective owners. No trademark rights are
licensed here. This notice describes independence; it is not trademark clearance.
