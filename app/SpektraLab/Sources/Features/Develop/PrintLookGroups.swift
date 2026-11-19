//  PrintLookGroups.swift — Film & Paper › Decode's print look, after the Enlarger: **Film Grain**, **Expand**,
//  **Print** and **Color Head**, each a row that folds on its own (as the Enlarger's), with its own ↺.
//
//  Controls after Dehancer's panels; the maths is our own, on the finished print before the output encoding
//  (`FilmParams.printLook`, the engine's `look_*`, `print_look.cpp`). Print layer: a slider re-prints.

import SwiftUI

struct PrintLookGroups: View {
    let session: OpenFrameSession

    var body: some View {
        VStack(spacing: 0) {
            PrintLookGroup(session: session, kind: .grain)
            PrintLookGroup(session: session, kind: .expand)
            PrintLookGroup(session: session, kind: .print)
            PrintLookGroup(session: session, kind: .head)
        }
    }
}

struct PrintLookGroup: View {
    enum Kind: String { case grain, expand, print, head }
    struct Row {
        let title: String
        let key: WritableKeyPath<PrintLook, Double>
        let range: ClosedRange<Double>
        let rest: Double
        let step: Double
        let decimals: Int
    }

    let session: OpenFrameSession
    let kind: Kind
    @AppStorage private var folded: Bool

    init(session: OpenFrameSession, kind: Kind) {
        self.session = session
        self.kind = kind
        _folded = AppStorage(wrappedValue: kind != .grain, "develop.printLook.\(kind.rawValue).folded")
    }

    private var look: PrintLook { session.edit.params.printLook }

    var title: String {
        switch kind {
        case .grain: L("Film Grain")
        case .expand: L("Expand")
        case .print: L("Print")
        case .head: L("Color Head")
        }
    }

    var rows: [Row] {
        switch kind {
        case .grain: [
            Row(title: L("Amount"), key: \.grainAmount, range: 0...100, rest: 0, step: 1, decimals: 0),
            Row(title: L("Size"), key: \.grainSize, range: 1...48, rest: 16, step: 0.5, decimals: 1),
            Row(title: L("Resolution"), key: \.grainResolution, range: 0...100, rest: 100, step: 1, decimals: 0),
            Row(title: L("Shadows"), key: \.grainShadows, range: 0...100, rest: 30, step: 1, decimals: 0),
            Row(title: L("Midtones"), key: \.grainMidtones, range: 0...100, rest: 55, step: 1, decimals: 0),
            Row(title: L("Highlights"), key: \.grainHighlights, range: 0...100, rest: 65, step: 1, decimals: 0),
            Row(title: L("Color"), key: \.grainColor, range: 0...100, rest: 65, step: 1, decimals: 0),
        ]
        case .expand: [
            Row(title: L("Black Point"), key: \.blackPoint, range: -20...30, rest: 0, step: 0.1, decimals: 2),
            Row(title: L("White Point"), key: \.whitePoint, range: 30...120, rest: 100, step: 0.1, decimals: 2),
        ]
        case .print: [
            Row(title: L("Exposure"), key: \.printExposure, range: -3...3, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Tonal Contrast"), key: \.tonalContrast, range: 0.5...2, rest: 1, step: 0.01, decimals: 2),
            Row(title: L("Color Density"), key: \.colorDensity, range: -100...100, rest: 0, step: 0.1, decimals: 2),
            Row(title: L("Saturation"), key: \.saturation, range: 0...200, rest: 100, step: 0.1, decimals: 2),
        ]
        case .head: [
            Row(title: L("Yellow - Blue"), key: \.headYellowBlue, range: -50...50, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Magenta - Green"), key: \.headMagentaGreen, range: -50...50, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Cyan - Red"), key: \.headCyanRed, range: -50...50, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Shadows Tone"), key: \.headShadows, range: -100...100, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Midtones Tone"), key: \.headMidtones, range: -100...100, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Highlights Tone"), key: \.headHighlights, range: -100...100, rest: 0, step: 0.01, decimals: 2),
            Row(title: L("Preserve Exposure"), key: \.headPreserve, range: 0...100, rest: 100, step: 0.1, decimals: 2),
            Row(title: L("Impact"), key: \.headImpact, range: 0...100, rest: 100, step: 0.1, decimals: 2),
        ]
        }
    }

    /// Why the look stands down, or nil where a print is made.
    private var reason: String? {
        if !session.edit.appliesFilm { return L("Off with film emulation off") }
        if session.edit.params.digitalIntermediate { return L("A Digital Intermediate is not printed") }
        return nil
    }

    private var moved: Bool { rows.contains { look[keyPath: $0.key] != $0.rest } }

    private func reset() {
        session.change { e in
            switch kind {
            case .grain: e.params.printLook.resetGrain()
            case .expand: e.params.printLook.resetExpand()
            case .print: e.params.printLook.resetPrint()
            case .head: e.params.printLook.resetHead()
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            ColumnSectionHeader(title: title, folded: folded, id: "printLook." + kind.rawValue,
                                rule: false, indent: HighDynamicRangeGroup.indent,
                                toggleFold: { folded.toggle() }) {
                RoundButton(diameter: 22, label: L("Reset") + " " + title, action: reset) {
                    Icon(.reset, size: 13)
                }
                .rowEnabled(reason == nil && moved)
                .accessibilityIdentifier("v5.develop.printLook.\(kind.rawValue).reset")
            }
            if !folded {
                if let reason {
                    Text(reason).font(Theme.Font.ui(11)).foregroundStyle(Theme.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16).padding(.bottom, 4)
                }
                ForEach(rows, id: \.title) { row in
                    ValueSlider(title: row.title,
                                value: session.binding((\FrameEdit.params.printLook).appending(path: row.key)),
                                range: row.range, rest: row.rest, step: row.step,
                                text: { String(format: "%.\(row.decimals)f", $0) },
                                detentAtRest: true, labelWidth: HighDynamicRangeGroup.labelWidth(row.title),
                                onEditingChanged: session.editing(.print))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 6)
                }
                .rowEnabled(reason == nil)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("v5.develop.printLook.\(kind.rawValue)")
    }
}
