from pydantic import BaseModel

# Models
class Applicant(BaseModel):

    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: str