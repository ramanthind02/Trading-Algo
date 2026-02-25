// Vertical line implementation for daily/weekly dividers

class VertLinePaneRenderer {
    constructor(x, options) {
        this._x = x;
        this._options = options;
    }

    draw(target) {
        target.useBitmapCoordinateSpace(scope => {
            if (this._x === null) return;
            const ctx = scope.context;
            const position = this.positionsLine(
                this._x,
                scope.horizontalPixelRatio,
                this._options.width
            );
            ctx.fillStyle = this._options.color;
            ctx.fillRect(
                position.position,
                0,
                position.length,
                scope.bitmapSize.height
            );
        });
    }

    positionsLine(positionMedia, pixelRatio, desiredWidthMedia = 1, widthIsBitmap) {
        const scaledPosition = Math.round(pixelRatio * positionMedia);
        const lineBitmapWidth = widthIsBitmap
            ? desiredWidthMedia
            : Math.round(desiredWidthMedia * pixelRatio);
        const offset = this.centreOffset(lineBitmapWidth);
        const position = scaledPosition - offset;
        return { position, length: lineBitmapWidth };
    }

    centreOffset(width) {
        return Math.floor(width / 2);
    }
}

class VertLinePaneView {
    constructor(source, options) {
        this._source = source;
        this._options = options;
    }

    update() {
        const timeScale = this._source._chart.timeScale();
        this._x = timeScale.timeToCoordinate(this._source._time);
    }

    renderer() {
        return new VertLinePaneRenderer(this._x, this._options);
    }
}

class VertLineTimeAxisView {
    constructor(source, options) {
        this._source = source;
        this._options = options;
    }

    update() {
        const timeScale = this._source._chart.timeScale();
        this._x = timeScale.timeToCoordinate(this._source._time);
    }

    visible() {
        return this._options.showLabel;
    }

    tickVisible() {
        return this._options.showLabel;
    }

    coordinate() {
        return this._x ?? 0;
    }

    text() {
        return this._options.labelText;
    }

    textColor() {
        return this._options.labelTextColor;
    }

    backColor() {
        return this._options.labelBackgroundColor;
    }
}

const vertLineDefaultOptions = {
    color: 'green',
    labelText: '',
    width: 2,
    labelBackgroundColor: 'green',
    labelTextColor: 'white',
    showLabel: false,
};

class VertLine {
    constructor(chart, series, time, options) {
        const vertLineOptions = {
            ...vertLineDefaultOptions,
            ...options,
        };
        this._chart = chart;
        this._series = series;
        this._time = time;
        this._paneViews = [new VertLinePaneView(this, vertLineOptions)];
        this._timeAxisViews = [new VertLineTimeAxisView(this, vertLineOptions)];
    }

    updateAllViews() {
        this._paneViews.forEach(pw => pw.update());
        this._timeAxisViews.forEach(tw => tw.update());
    }

    timeAxisViews() {
        return this._timeAxisViews;
    }

    paneViews() {
        return this._paneViews;
    }
}
