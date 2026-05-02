#include "Screen.h"
#include "daisy_pod.h"
#include <cstdint>
#include <cstdio>
using namespace daisy;

extern DaisyPod hw;

static SpiHandle spi;
static GPIO tftDc;
static GPIO tftRst;

#define ILI9341_SWRESET 0x01
#define ILI9341_SLPOUT  0x11
#define ILI9341_DISPON  0x29
#define ILI9341_CASET   0x2A
#define ILI9341_PASET   0x2B
#define ILI9341_RAMWR   0x2C
#define ILI9341_MADCTL  0x36
#define ILI9341_PIXFMT  0x3A

#define BLACK   0x0000
#define WHITE   0xFFFF
#define RED     0xF800
#define GREEN   0x07E0
#define BLUE    0x001F
#define YELLOW  0xFFE0
#define CYAN    0x07FF

static const uint8_t font5x7[96][5] = {
    {0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},{0x14,0x7F,0x14,0x7F,0x14},
    {0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},{0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},
    {0x00,0x1C,0x22,0x41,0x00},{0x00,0x41,0x22,0x1C,0x00},{0x14,0x08,0x3E,0x08,0x14},{0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},{0x20,0x10,0x08,0x04,0x02},
    {0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},{0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},
    {0x18,0x14,0x12,0x7F,0x10},{0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},{0x00,0x56,0x36,0x00,0x00},
    {0x08,0x14,0x22,0x41,0x00},{0x14,0x14,0x14,0x14,0x14},{0x00,0x41,0x22,0x14,0x08},{0x02,0x01,0x51,0x09,0x06},
    {0x32,0x49,0x79,0x41,0x3E},{0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x09,0x01},{0x3E,0x41,0x49,0x49,0x7A},
    {0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},{0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},
    {0x7F,0x40,0x40,0x40,0x40},{0x7F,0x02,0x0C,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},{0x46,0x49,0x49,0x49,0x31},
    {0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},{0x1F,0x20,0x40,0x20,0x1F},{0x7F,0x20,0x18,0x20,0x7F},
    {0x63,0x14,0x08,0x14,0x63},{0x03,0x04,0x78,0x04,0x03},{0x61,0x51,0x49,0x45,0x43},{0x00,0x7F,0x41,0x41,0x00},
    {0x02,0x04,0x08,0x10,0x20},{0x00,0x41,0x41,0x7F,0x00},{0x04,0x02,0x01,0x02,0x04},{0x40,0x40,0x40,0x40,0x40},
    {0x00,0x01,0x02,0x04,0x00},{0x20,0x54,0x54,0x54,0x78},{0x7F,0x48,0x44,0x44,0x38},{0x38,0x44,0x44,0x44,0x20},
    {0x38,0x44,0x44,0x48,0x7F},{0x38,0x54,0x54,0x54,0x18},{0x08,0x7E,0x09,0x01,0x02},{0x0C,0x52,0x52,0x52,0x3E},
    {0x7F,0x08,0x04,0x04,0x78},{0x00,0x44,0x7D,0x40,0x00},{0x20,0x40,0x44,0x3D,0x00},{0x7F,0x10,0x28,0x44,0x00},
    {0x00,0x41,0x7F,0x40,0x00},{0x7C,0x04,0x18,0x04,0x78},{0x7C,0x08,0x04,0x04,0x78},{0x38,0x44,0x44,0x44,0x38},
    {0x7C,0x14,0x14,0x14,0x08},{0x08,0x14,0x14,0x18,0x7C},{0x7C,0x08,0x04,0x04,0x08},{0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3F,0x44,0x40,0x20},{0x3C,0x40,0x40,0x20,0x7C},{0x1C,0x20,0x40,0x20,0x1C},{0x3C,0x40,0x30,0x40,0x3C},
    {0x44,0x28,0x10,0x28,0x44},{0x0C,0x50,0x50,0x50,0x3C},{0x44,0x64,0x54,0x4C,0x44},{0x00,0x08,0x36,0x41,0x00},
    {0x00,0x00,0x7F,0x00,0x00},{0x00,0x41,0x36,0x08,0x00},{0x10,0x08,0x08,0x10,0x08},{0x00,0x00,0x00,0x00,0x00}
};

static void DelayMs(uint32_t ms)
{
    System::Delay(ms);
}

static void DcCommand()
{
    tftDc.Write(false);
}

static void DcData()
{
    tftDc.Write(true);
}

static void SpiWrite(uint8_t data)
{
    spi.BlockingTransmit(&data, 1);
}

static void WriteCommand(uint8_t cmd)
{
    DcCommand();
    SpiWrite(cmd);
}

static void WriteData(uint8_t data)
{
    DcData();
    SpiWrite(data);
}

static void ResetDisplay()
{
    tftRst.Write(false);
    DelayMs(100);
    tftRst.Write(true);
    DelayMs(150);
}

static void SetAddressWindow(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    WriteCommand(ILI9341_CASET);
    WriteData(x0 >> 8);
    WriteData(x0 & 0xFF);
    WriteData(x1 >> 8);
    WriteData(x1 & 0xFF);

    WriteCommand(ILI9341_PASET);
    WriteData(y0 >> 8);
    WriteData(y0 & 0xFF);
    WriteData(y1 >> 8);
    WriteData(y1 & 0xFF);

    WriteCommand(ILI9341_RAMWR);
}

static void FillRect(int x, int y, int w, int h, uint16_t colour)
{
    if(x < 0 || y < 0 || w <= 0 || h <= 0) return;
    if(x + w > 240) w = 240 - x;
    if(y + h > 320) h = 320 - y;

    SetAddressWindow(x, y, x + w - 1, y + h - 1);
    DcData();

    uint8_t hi = colour >> 8;
    uint8_t lo = colour & 0xFF;

    for(int i = 0; i < w * h; i++)
    {
        uint8_t data[2] = {hi, lo};
        spi.BlockingTransmit(data, 2);
    }
}

static void FillScreen(uint16_t colour)
{
    FillRect(0, 0, 240, 320, colour);
}

static void DrawPixel(int x, int y, uint16_t colour)
{
    FillRect(x, y, 1, 1, colour);
}

static void DrawChar(int x, int y, char c, uint16_t colour, uint16_t bg, int scale)
{
    if(c < 32 || c > 127){
        c = '?';
    }

    const uint8_t* bitmap = font5x7[c - 32];

    for(int col = 0; col < 5; col++){
        uint8_t line = bitmap[col];

        for(int row = 0; row < 7; row++){
            uint16_t pixelColour = (line & (1 << row)) ? colour : bg;

            if(scale <= 1){
                DrawPixel(x + col, y + row, pixelColour);
            }
            else{
                FillRect(x + col * scale, y + row * scale, scale, scale, pixelColour);
            }
        }
    }

    if(scale <= 1){
        FillRect(x + 5, y, 1, 7, bg);
    }
    else{
        FillRect(x + 5 * scale, y, scale, 7 * scale, bg);
    }
}

static void DrawTextLine(int x, int y, const char* text, uint16_t colour)
{
    int cursor = x;
    int scale = 2;

    while(*text && cursor < 230){
        DrawChar(cursor, y, *text, colour, BLACK, scale);
        cursor += 6 * scale;
        text++;
    }
}

static void DrawTextSmall(int x, int y, const char* text, uint16_t colour)
{
    int cursor = x;
    int scale = 1;

    while(*text && cursor < 235){
        DrawChar(cursor, y, *text, colour, BLACK, scale);
        cursor += 6;
        text++;
    }
}

static void DrawHeader(const char* title)
{
    FillRect(0, 0, 240, 32, BLUE);
    int cursor = 10;

    while(*title && cursor < 230){
        DrawChar(cursor, 9, *title, WHITE, BLUE, 2);
        cursor += 12;
        title++;
    }
}

static void InitDisplay()
{
    ResetDisplay();

    WriteCommand(ILI9341_SWRESET);
    DelayMs(150);

    WriteCommand(ILI9341_SLPOUT);
    DelayMs(150);

    WriteCommand(0xCF); WriteData(0x00); WriteData(0xC1); WriteData(0x30);
    WriteCommand(0xED); WriteData(0x64); WriteData(0x03); WriteData(0x12); WriteData(0x81);
    WriteCommand(0xE8); WriteData(0x85); WriteData(0x00); WriteData(0x78);
    WriteCommand(0xCB); WriteData(0x39); WriteData(0x2C); WriteData(0x00); WriteData(0x34); WriteData(0x02);
    WriteCommand(0xF7); WriteData(0x20);
    WriteCommand(0xEA); WriteData(0x00); WriteData(0x00);

    WriteCommand(0xC0); WriteData(0x23);
    WriteCommand(0xC1); WriteData(0x10);
    WriteCommand(0xC5); WriteData(0x3E); WriteData(0x28);
    WriteCommand(0xC7); WriteData(0x86);

    WriteCommand(ILI9341_MADCTL);
    WriteData(0x48);

    WriteCommand(ILI9341_PIXFMT);
    WriteData(0x55);

    WriteCommand(ILI9341_DISPON);
    DelayMs(150);
}

void ScreenInit()
{
    tftDc.Init(seed::A1, GPIO::Mode::OUTPUT);
    tftRst.Init(seed::A7, GPIO::Mode::OUTPUT);

    tftDc.Write(true);
    tftRst.Write(true);

    SpiHandle::Config spiConfig;
    spiConfig.periph = SpiHandle::Config::Peripheral::SPI_1;
    spiConfig.mode = SpiHandle::Config::Mode::MASTER;
    spiConfig.direction = SpiHandle::Config::Direction::TWO_LINES;
    spiConfig.datasize = 8;
    spiConfig.clock_polarity = SpiHandle::Config::ClockPolarity::LOW;
    spiConfig.clock_phase = SpiHandle::Config::ClockPhase::ONE_EDGE;
    spiConfig.nss = SpiHandle::Config::NSS::SOFT;
    spiConfig.baud_prescaler = SpiHandle::Config::BaudPrescaler::PS_64;

    spiConfig.pin_config.sclk = seed::D8;
    spiConfig.pin_config.miso = seed::D9;
    spiConfig.pin_config.mosi = seed::D10;
    spiConfig.pin_config.nss  = Pin();

    spi.Init(spiConfig);

    InitDisplay();
    FillScreen(BLACK);
}

void ScreenFillRed()
{
    FillScreen(RED);
}

void ScreenFillBlack()
{
    FillScreen(BLACK);
}

void ScreenDrawHome(const char* presetName,
                    const char* networkName,
                    int isTraining,
                    int predictionMode,
                    int predictedClass,
                    int stableClass)
{
    FillScreen(BLACK);

    DrawHeader("PEDAL");

    DrawTextSmall(10, 45, "Preset:", CYAN);
    DrawTextSmall(85, 45, presetName, WHITE);

    DrawTextSmall(10, 70, "Network:", CYAN);
    DrawTextSmall(85, 70, networkName, WHITE);

    DrawTextSmall(10, 95, "Training:", CYAN);
    DrawTextSmall(95, 95, isTraining ? "ON" : "OFF", isTraining ? GREEN : RED);

    DrawTextSmall(10, 120, "Predict:", CYAN);
    DrawTextSmall(95, 120, predictionMode ? "ON" : "OFF", predictionMode ? GREEN : RED);

    DrawTextSmall(10, 150, "Pred:", YELLOW);
    if(predictedClass == 0) DrawTextSmall(75, 150, "Distortion", WHITE);
    if(predictedClass == 1) DrawTextSmall(75, 150, "Chorus", WHITE);
    if(predictedClass == 2) DrawTextSmall(75, 150, "Ambient", WHITE);
    if(predictedClass == 3) DrawTextSmall(75, 150, "Reverb", WHITE);

    DrawTextSmall(10, 175, "Stable:", YELLOW);
    if(stableClass == 0) DrawTextSmall(75, 175, "Distortion", WHITE);
    if(stableClass == 1) DrawTextSmall(75, 175, "Chorus", WHITE);
    if(stableClass == 2) DrawTextSmall(75, 175, "Ambient", WHITE);
    if(stableClass == 3) DrawTextSmall(75, 175, "Reverb", WHITE);
}

void ScreenDrawMenu(int selectedIndex,
                    const char* presetName,
                    const char* networkName,
                    int isTraining,
                    int predictionMode,
                    int hasSavedMapping,
                    int b0,
                    int b1,
                    int b2,
                    int b3)
{
    static bool firstDraw = true;
    static int lastSelectedIndex = -1;
    static int lastB0 = -1;
    static int lastB1 = -1;
    static int lastB2 = -1;
    static int lastB3 = -1;

    const char* items[5] = {
        "Preset:",
        "Network:",
        "Training:",
        "Save Map",
        "Predict:"
    };

    const char* values[5] = {
        presetName,
        networkName,
        isTraining ? "ON" : "OFF",
        hasSavedMapping ? "Saved" : "Train",
        predictionMode ? "ON" : "OFF"
    };

    if(firstDraw){
        FillScreen(BLACK);
        DrawHeader("MENU");
        firstDraw = false;
        lastSelectedIndex = -1;
    }

    for(int i = 0; i < 5; i++){
        bool redrawRow =
            i == selectedIndex ||
            i == lastSelectedIndex ||
            lastSelectedIndex == -1;

        if(redrawRow){
            int y = 50 + i * 38;
            uint16_t bg = i == selectedIndex ? BLUE : BLACK;

            FillRect(0, y - 8, 240, 28, bg);

            if(i == selectedIndex){
                DrawChar(8, y, '>', YELLOW, BLUE, 2);
            }

            int cursor = 32;
            const char* p = items[i];

            while(*p && cursor < 125){
                DrawChar(cursor, y, *p, WHITE, bg, 2);
                cursor += 12;
                p++;
            }

            cursor = 140;
            p = values[i];

            while(*p && cursor < 235){
                DrawChar(cursor, y, *p, CYAN, bg, 2);
                cursor += 12;
                p++;
            }
        }
    }

    if(b0 != lastB0 || b1 != lastB1 || b2 != lastB2 || b3 != lastB3){
        FillRect(0, 250, 240, 35, BLACK);

        DrawTextSmall(10, 260, "Buffers:", YELLOW);

        char buf[32];
        snprintf(buf, sizeof(buf), "D%d C%d A%d R%d", b0, b1, b2, b3);
        DrawTextSmall(80, 260, buf, WHITE);

        lastB0 = b0;
        lastB1 = b1;
        lastB2 = b2;
        lastB3 = b3;
    }

    lastSelectedIndex = selectedIndex;
}