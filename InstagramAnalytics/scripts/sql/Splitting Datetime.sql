ALTER TABLE AccountSnapshot
ADD
    RetrievedDate AS CAST(RetrievedAt AS DATE) PERSISTED,
    RetrievedTime AS CAST(RetrievedAt AS TIME) PERSISTED;
GO

ALTER TABLE Post
ADD
    PostDate AS CAST([Date] AS DATE) PERSISTED,
    PostTime AS CAST([Date] AS TIME) PERSISTED;

GO