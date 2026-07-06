import os
import uvicorn
from scrinium import app


def main():
    port = int(os.getenv("PORT", "9231"))
    uvicorn.run("scrinium:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
