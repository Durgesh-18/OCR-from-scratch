var ocrDemo = {
    // ---- CONFIG ----
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10,
    BLUE: "#cccccc",

    HOST: "http://localhost",
    PORT: 8000,
    BATCH_SIZE: 10,

    // ---- STATE ----
    data: [],
    trainArray: [],
    trainingRequestCount: 0,
    canvas: null,
    ctx: null,

    // ---- INIT (CALLED FROM HTML onload) ----
    onLoadFunction: function () {
        // initialize pixel data
        this.data = new Array(
            this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH
        ).fill(0);

        this.trainArray = [];
        this.trainingRequestCount = 0;

        // get canvas and context
        this.canvas = document.getElementById("canvas");
        this.ctx = this.canvas.getContext("2d");

        // draw grid
        this.drawGrid(this.ctx);

        // mouse events
        this.canvas.onmousedown = (e) => {
            this.onMouseDown(e, this.ctx, this.canvas);
        };

        this.canvas.onmousemove = (e) => {
            this.onMouseMove(e, this.ctx, this.canvas);
        };

        this.canvas.onmouseup = () => {
            this.canvas.isDrawing = false;
        };
    },

    // ---- DRAWING ----
    drawGrid: function (ctx) {
        ctx.clearRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);

        for (let i = 0; i <= this.CANVAS_WIDTH; i += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;

            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(this.CANVAS_WIDTH, i);
            ctx.stroke();
        }
    },

    fillSquare: function (ctx, x, y) {
        let xPixel = Math.floor(x / this.PIXEL_WIDTH);
        let yPixel = Math.floor(y / this.PIXEL_WIDTH);

        let index = yPixel * this.TRANSLATED_WIDTH + xPixel;
        if (index < 0 || index >= this.data.length) return;

        this.data[index] = 1;

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(
            xPixel * this.PIXEL_WIDTH,
            yPixel * this.PIXEL_WIDTH,
            this.PIXEL_WIDTH,
            this.PIXEL_WIDTH
        );
    },

    onMouseDown: function (e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(
            ctx,
            e.clientX - canvas.offsetLeft,
            e.clientY - canvas.offsetTop
        );
    },

    onMouseMove: function (e, ctx, canvas) {
        if (!canvas.isDrawing) return;
        this.fillSquare(
            ctx,
            e.clientX - canvas.offsetLeft,
            e.clientY - canvas.offsetTop
        );
    },

    // ---- ACTIONS ----
    resetCanvas: function () {
        this.data.fill(0);
        this.drawGrid(this.ctx);
    },

    train: function () {
        let digitVal = document.getElementById("digit").value;

        if (!digitVal || this.data.indexOf(1) === -1) {
            alert("Please draw a digit and enter its value.");
            return;
        }

        this.trainArray.push({
            y0: this.data.slice(),
            label: parseInt(digitVal)
        });

        this.trainingRequestCount++;

        if (this.trainingRequestCount === this.BATCH_SIZE) {
            let json = {
                train: true,
                trainArray: this.trainArray
            };

            this.sendData(json);

            this.trainingRequestCount = 0;
            this.trainArray = [];
        }

        this.resetCanvas();
    },

    test: function () {
        if (this.data.indexOf(1) === -1) {
            alert("Please draw a digit first.");
            return;
        }

        let json = {
            predict: true,
            image: this.data
        };

        this.sendData(json);
    },

    // ---- NETWORK ----
    sendData: function (json) {
    let xhr = new XMLHttpRequest();
    xhr.open("POST", this.HOST + ":" + this.PORT, true);
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.onreadystatechange = () => {
        if (xhr.readyState !== XMLHttpRequest.DONE) return;

        // Server reachable, even if error
        if (xhr.status >= 200 && xhr.status < 300) {
            this.receiveResponse(xhr);
        } else {
            // Try to show backend error message
            try {
                const err = JSON.parse(xhr.responseText);
                alert("Server error: " + err.message);
            } catch {
                alert("Server error (status " + xhr.status + ")");
            }
        }
    };

    xhr.onerror = () => {
        alert("Network error: server unreachable");
    };

    xhr.send(JSON.stringify(json));
},

    receiveResponse: function (xhr) {
        let response = JSON.parse(xhr.responseText);

        if (response.type === "test") {
            alert("Prediction: " + response.result);
        }
    }
};
